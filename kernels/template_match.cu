#include <torch/extension.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>
#include <thrust/pair.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include <nvtx3/nvToolsExt.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <limits>


__global__ void templateMatchKernel
(
	const float* __restrict__ input,const float* templ,float* output,
	int img_w,int img_h,
	int tpl_w, int tpl_h
) 
{
	int res_w = img_w - tpl_w + 1;
	int res_h = img_h - tpl_h + 1;
	int pad_x = tpl_w / 2;
	int pad_y = tpl_h / 2; 
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//load template into shared memory for faster access
	extern __shared__ float shared_tpl[];
	int template_id = threadIdx.y * blockDim.x + threadIdx.x;
	int template_size = tpl_w * tpl_h;
	for (int i = template_id; i < template_size; i += blockDim.x * blockDim.y) {
		shared_tpl[i] = templ[i]; //blue channel
		shared_tpl[i + tpl_w * tpl_h] = templ[i + tpl_w * tpl_h]; //green channel
		shared_tpl[i + 2 * tpl_w * tpl_h] = templ[i + 2 * tpl_w * tpl_h]; //red channel
		shared_tpl[i + 3 * tpl_w * tpl_h] = templ[i + 3 * tpl_w * tpl_h]; //alpha channel

	}
	__syncthreads();


	// corelation for each pixel
	if (x < res_w && y < res_h) {
		float sum = 0.0f;
		for (int i = 0; i < tpl_w; i++) {
			for (int j = 0; j < tpl_h; j++) {
				//we calculate diff only for pixels where template alpha > 0.5
				float tpl_val_alpha = shared_tpl[j * tpl_w + i + 3 * tpl_w * tpl_h];
				if (tpl_val_alpha < 0.5) continue;

				// if we flatten the image we get y*width + x so for correlation we offset by i and j thus
				float img_val_blue = input[(y + j) * img_w + x + i];
				float img_val_green = input[(y + j) * img_w + x + i + img_w * img_h];
				float img_val_red = input[(y + j) * img_w + x + i + 2 * img_w * img_h];

				// get template value simply y_tpl*width_tpl + x_tpl 
				float tpl_val_blue = shared_tpl[j * tpl_w + i];
				float tpl_val_green = shared_tpl[j * tpl_w + i + tpl_w * tpl_h];
				float tpl_val_red = shared_tpl[j * tpl_w + i + 2 * tpl_w * tpl_h];


				float diff_r = img_val_red - tpl_val_red;
				float diff_g = img_val_green - tpl_val_green;
				float diff_b = img_val_blue - tpl_val_blue;
				sum += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;//TM_SQ_DIFF
			}
		}
		//same as before
		int out_idx = (y + pad_y) * img_w + (x + pad_x);
		output[out_idx] = sum;
	}

}
std::tuple<int,int> template_match(torch::Tensor img, torch::Tensor templ) {
	//check if device is cuda and type is float
	TORCH_CHECK(img.device().type() == torch::kCUDA);
	TORCH_CHECK(img.dtype() == torch::kFloat);

	int img_h = img.size(1);
	int img_w = img.size(2);
	int img_ch = 3; //RGB
 	int tpl_h = templ.size(1);
	int tpl_w = templ.size(2);
	int tpl_ch = 4; //RGBA
	int res_w = img_w - tpl_w + 1;
	int res_h = img_h - tpl_h + 1;
	int pad_x = tpl_w / 2;
	int pad_y = tpl_h / 2;
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(img.device());
	auto result = torch::full({ img_h,img_w }, std::numeric_limits<float>::infinity(), options);
	size_t shared_mem_size = 4 * tpl_w * tpl_h * sizeof(float); //4 channels

	dim3 dimBlock = getOptimalBlockDim(res_w, res_h);
	dim3 dimGrid(cdiv(res_w, dimBlock.x), cdiv(res_h, dimBlock.y));

	//flatten permuted tensors for easy indexing
	torch::Tensor img_flat = img.flatten();
	torch::Tensor templ_flat = templ.flatten();


	cudaStream_t cub_stream, thrust_stream;
	cudaStreamCreate(&cub_stream);
	cudaStreamCreate(&thrust_stream);

	cudaEvent_t kernel_finished;
	cudaEventCreate(&kernel_finished);


	nvtxRangePushA("Template_Match_Kernel");
	templateMatchKernel <<< dimGrid, dimBlock, shared_mem_size, cub_stream >>> (
		img_flat.data_ptr<float>(),
		templ_flat.data_ptr<float>(),
		result.data_ptr<float>(),
		img_w, img_h,
		tpl_w, tpl_h
		);
	nvtxRangePop();

	cudaEventRecord(kernel_finished, cub_stream);
	cudaStreamWaitEvent(thrust_stream, kernel_finished, 0);

	//----allocating CUB resources----
	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::KeyValuePair<int, float>* d_out;
	cudaMalloc(&d_out, sizeof(cub::KeyValuePair<int, float>));
	const float* d_in = result.data_ptr<float>();

	cub::DeviceReduce::ArgMin(
		d_temp_storage,
		temp_storage_bytes,
		d_in,
		d_out,
		img_w * img_h,
		cub_stream
	);
	//allocating temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::KeyValuePair<int, float>* h_out;
	cudaMallocHost(&h_out, sizeof(cub::KeyValuePair<int, float>));


	//----allocating THRUST resources----
	float* result_ptr = result.data_ptr<float>();
	//pair of value and index
	auto extract_value = [=] __host__ __device__(int idx) -> thrust::pair<float, int> {
		int x = idx % res_w;
		int y = idx / res_w;
		int out_idx = (y + pad_y) * img_w + (x + pad_x);
		return thrust::pair<float, int>(result_ptr[out_idx], out_idx);

	};

	auto value_iter = thrust::make_transform_iterator(
		thrust::counting_iterator<int>(0),
		extract_value
	);

	auto thrust_policy = thrust::cuda::par.on(thrust_stream);


	//-----------------FINDING MINIMUM CUB WAY--------------------
	nvtxRangePushA("Find_Minimum_CUB");
	cub::DeviceReduce::ArgMin(
		d_temp_storage,
		temp_storage_bytes,
		d_in,
		d_out,
		img_w * img_h,
		cub_stream
	);
	cudaMemcpyAsync(h_out, d_out, sizeof(cub::KeyValuePair<int, float>), cudaMemcpyDeviceToHost, cub_stream);
	nvtxRangePop();

	// ------------------------ FINDING MINIMUM THRUST WAY-----------------
	nvtxRangePushA("Find_Minimum_thrust");
	//find minimum value and its index
	auto min_result_thrust = thrust::reduce(
		thrust_policy,
		value_iter,
		value_iter + (res_w * res_h),
		thrust::pair<float, int>(std::numeric_limits<float>::infinity(), -1),
		[]__host__ __device__(thrust::pair<float, int> a, thrust::pair<float, int>b) {
		return (a.first > b.first) ? b : a;
	}
	);

	nvtxRangePop();

	//reading from thrust result
	cudaStreamSynchronize(thrust_stream);
	float min_value_thrust = min_result_thrust.first;
	int min_index_thrust = min_result_thrust.second;
	int min_x_thrust = min_index_thrust % img_w;
	int min_y_thrust = min_index_thrust / img_w;
	printf("THRUST SQ_DIFF min val:%f at (x = %d,y = %d)\n", min_value_thrust, min_x_thrust, min_y_thrust);

	//reading from CUB result
	cudaStreamSynchronize(cub_stream);
	float min_value_cub = h_out->value;
	int min_index_cub = h_out->key;
	int min_x_cub = min_index_cub % img_w;
	int min_y_cub = min_index_cub / img_w;
	printf("CUB SQ_DIFF min val:%f at (x = %d,y = %d)\n", min_value_cub, min_x_cub, min_y_cub);

	cudaFree(d_out);
	cudaFree(d_temp_storage);
	cudaFreeHost(h_out);
	cudaStreamDestroy(cub_stream);
	cudaStreamDestroy(thrust_stream);
	cudaEventDestroy(kernel_finished);

	return std::tuple<int, int>(min_x_cub, min_y_cub);

}
