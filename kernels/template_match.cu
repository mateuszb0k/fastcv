#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include <torch/extension.h>
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
	// corelation for each pixel
	if (x < res_w && y < res_h) {
		float sum = 0.0f;
		for (int i = 0; i < tpl_w; i++) {
			for (int j = 0; j < tpl_h; j++) {
				// if we flatten the image we get y*width + x so for correlation we offset by i and j thus
				float img_val = input[(y + j) * img_w + x + i];
				// get template value simply y_tpl*width_tpl + x_tpl 
				float tpl_val = templ[j * tpl_w + i];
				float diff = img_val - tpl_val;
				sum += diff * diff;//TM_SQ_DIFF
			}
		}
		//same as before
		int out_idx = (y + pad_y) * img_w + (x + pad_x);
		output[out_idx] = sum;
	}

}
torch::Tensor template_match(torch::Tensor img, torch::Tensor templ) {
	//check if device is cuda and type is byte stolen from sobel.cu
	TORCH_CHECK(img.device().type() == torch::kCUDA);
	TORCH_CHECK(img.dtype() == torch::kFloat);
	int img_h = img.size(0);
	int img_w = img.size(1);
	int tpl_h = templ.size(0);
	int tpl_w = templ.size(1);
	int res_w = img_w - tpl_w + 1;
	int res_h = img_h - tpl_h + 1;
	auto result = torch::full_like(img,1000000.0f);
	dim3 dimBlock = getOptimalBlockDim(res_w, res_h);
	dim3 dimGrid(cdiv(res_w, dimBlock.x), cdiv(res_h, dimBlock.y));
	templateMatchKernel <<< dimGrid, dimBlock >>> (
		img.data_ptr<float>(),
		templ.data_ptr<float>(),
		result.data_ptr<float>(),
		img_w, img_h,
		tpl_w, tpl_h

		);
	cudaDeviceSynchronize();
	return result;

}
