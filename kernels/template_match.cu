#include <torch/extension.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>
#include <thrust/pair.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include <nvtx3/nvToolsExt.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
//type aliases 
//modern cpp requirements
//using DeviceVec = thrust::device_vector<float>;
//using HostVec = thrust::host_vector<float>;
//using UniversalVec = thrust::universal_vector<float>;
// 
//using KeyValuePair = cub::KeyValuePair<int, float>;

__global__ void templateMatchKernel
(
	const float* input,const float* templ,float* output,
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
		shared_tpl[i] = templ[i];
	}
	__syncthreads();


	// corelation for each pixel
	if (x < res_w && y < res_h) {
		float sum = 0.0f;
		for (int i = 0; i < tpl_w; i++) {
			for (int j = 0; j < tpl_h; j++) {
				// if we flatten the image we get y*width + x so for correlation we offset by i and j thus
				float img_val = input[(y + j) * img_w + x + i];
				// get template value simply y_tpl*width_tpl + x_tpl 
				float tpl_val = shared_tpl[j * tpl_w + i];
				float diff = img_val - tpl_val;
				sum += diff * diff;//TM_SQ_DIFF
			}
		}
		//same as before
		int out_idx = (y + pad_y) * img_w + (x + pad_x);
		output[out_idx] = sum;
	}

}
std::tuple<int,int> template_match(torch::Tensor img, torch::Tensor templ) {
	//check if device is cuda and type is byte stolen from sobel.cu
	TORCH_CHECK(img.device().type() == torch::kCUDA);
	TORCH_CHECK(img.dtype() == torch::kFloat);
	int img_h = img.size(0);
	int img_w = img.size(1);
	int tpl_h = templ.size(0);
	int tpl_w = templ.size(1);
	int res_w = img_w - tpl_w + 1;
	int res_h = img_h - tpl_h + 1;
	int pad_x = tpl_w / 2;
	int pad_y = tpl_h / 2;
	auto result = torch::full_like(img,1000000.0f);
	size_t shared_mem_size = tpl_w * tpl_h * sizeof(float);

	dim3 dimBlock = getOptimalBlockDim(res_w, res_h);
	dim3 dimGrid(cdiv(res_w, dimBlock.x), cdiv(res_h, dimBlock.y));

	nvtxRangePushA("Template_Match_Kernel");
	templateMatchKernel <<< dimGrid, dimBlock, shared_mem_size >>> (
		img.data_ptr<float>(),
		templ.data_ptr<float>(),
		result.data_ptr<float>(),
		img_w, img_h,
		tpl_w, tpl_h

		);
	nvtxRangePop();

	cudaDeviceSynchronize();
	float* result_ptr = result.data_ptr<float>();
	//pair of value and index
	auto extract_value = [=] __host__ __device__(int idx) -> thrust::pair<float, int> {
		int x = idx % res_w;
		int y = idx / res_w;
		int out_idx = (y + pad_y) * img_w + (x + pad_x);
		return thrust::pair<float, int>(result_ptr[out_idx], out_idx);

	};


	// ------------------------ FINDING MINIMUM THRUST WAY-----------------
	//iterate over all result values
	nvtxRangePushA("Find_Minimum_thrust");
	auto value_iter = thrust::make_transform_iterator(
		thrust::counting_iterator<int>(0),
		extract_value
	);

	//find minimum value and its index
	auto min_result = thrust::reduce(
		thrust::device,
		value_iter,
		value_iter + (res_w * res_h),
		thrust::pair<float, int>(1e9f, -1),
		[]__host__ __device__(thrust::pair<float, int> a, thrust::pair<float, int>b) {
		return (a.first > b.first) ? b : a;
	}
	);

	float min_value = min_result.first;
	int min_index = min_result.second;
	int min_x = min_index % img_w;
	int min_y = min_index / img_w;
	printf("THRUST SQ_DIFF min val:%f at (x = %d,y = %d)\n", min_value, min_x, min_y);

	nvtxRangePop();

	//-----------------FINDING MINIMUM CUB WAY--------------------
	nvtxRangePushA("Find_Minimum_CUB");
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
		img_w * img_h
	);
	//allocating temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	cub::DeviceReduce::ArgMin(
		d_temp_storage,
		temp_storage_bytes,
		d_in,
		d_out,
		img_w * img_h
	);

	cub::KeyValuePair<int, float> h_out;
	cudaMemcpy(&h_out, d_out, sizeof(cub::KeyValuePair<int, float>), cudaMemcpyDeviceToHost);

	min_value = h_out.value;
	min_index = h_out.key;
	min_x = min_index % img_w;
	min_y = min_index / img_w;
	printf("CUB SQ_DIFF min val:%f at (x = %d,y = %d)\n", min_value, min_x, min_y);
	cudaFree(d_out);
	cudaFree(d_temp_storage);

	nvtxRangePop();


	return std::tuple<int, int>(min_x, min_y);

}
