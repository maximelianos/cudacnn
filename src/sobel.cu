/*
 ============================================================================
 Name        : sobel.cu
 Author      : maximelianos
 Version     :
 Copyright   : Created by maximelianos, all rights reserved.
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <math.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>

#include <omp.h>

#include "imageLoader.h"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define MAX_THREADS 128

////////////////////////////////////////////////////////////////

byte *alloc_cuda_zero_b(unsigned int size) {
	byte *res;
	CUDA_CHECK_RETURN(
		cudaMalloc( (void**)&res, (size * sizeof(*res))) );
	CUDA_CHECK_RETURN(
		cudaMemset(res, 0, (size*sizeof(*res))) );
	return res;
}

float *alloc_cuda_zero_f(unsigned int size) {
	float *res;
	CUDA_CHECK_RETURN(
		cudaMalloc( (void**)&res, (size * sizeof(float))) );
	CUDA_CHECK_RETURN(
		cudaMemset(res, 0, (size*sizeof(float))) );
	return res;
}

byte *alloc_cuda_copy_host_b(byte *input, unsigned int size) {
	byte *res;
	CUDA_CHECK_RETURN(
		cudaMalloc( (void**)&res, (size * sizeof(*res))) );
	CUDA_CHECK_RETURN(
		cudaMemcpy(res,
				   input,
				   (size * sizeof(*res)),
				   cudaMemcpyHostToDevice)
		);
	return res;
}

float *alloc_cuda_copy_host_f(float *input, unsigned int size) {
	float *res;
	CUDA_CHECK_RETURN(
		cudaMalloc( (void**)&res, (size * sizeof(float))) );
	CUDA_CHECK_RETURN(
		cudaMemcpy(res,
				   input,
				   (size * sizeof(float)),
				   cudaMemcpyHostToDevice)
		);
	return res;
}

float *alloc_cuda_copy_device(float *input, unsigned int size) {
	float *res;
	CUDA_CHECK_RETURN(
		cudaMalloc( (void**)&res, (size * sizeof(float))) );
	CUDA_CHECK_RETURN(
		cudaMemcpy(res,
				   input,
				   (size * sizeof(float)),
				   cudaMemcpyDeviceToDevice)
		);
	return res;
}

////////////////////////////////////////////////////////////////


// Feature values with shape H * W * C stored on GPU
struct feature {
	float *f;
	unsigned int h, w, c;
};

// Convolution kernel weights with shape H * W * input_c * output_c
struct convlayer {
	// f = sum(input * w) + b
	float *w; // weights
	float *b;  // bias
	unsigned int n;
};

// Fixed Sobel kernel
__global__ void sobel_gpu(const byte* orig, byte* result, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
        if( x > 0 && y > 0 && x < width-1 && y < height-1) {
            dx = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
                 (    orig[(y-1)*width + (x+1)]) + ( 2*orig[y*width+(x+1)]) + (   orig[(y+1)*width+(x+1)]);
            dy = (    orig[(y-1)*width + (x-1)]) + ( 2*orig[(y-1)*width+x]) + (   orig[(y-1)*width+(x+1)]) +
                 (-1* orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
            result[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
        }
}

// Sobel with new feature structure in args
__global__ void sobel_f(const struct feature input, struct feature output) {
	int height = input.h;
	int width = input.w;
	float *f = input.f;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
        if( x > 0 && y > 0 && x < width-1 && y < height-1) {
            dx = (-1* f[(y-1)*width + (x-1)]) + (-2*f[y*width+(x-1)]) + (-1*f[(y+1)*width+(x-1)]) +
                 (    f[(y-1)*width + (x+1)]) + ( 2*f[y*width+(x+1)]) + (   f[(y+1)*width+(x+1)]);
            dy = (    f[(y-1)*width + (x-1)]) + ( 2*f[(y-1)*width+x]) + (   f[(y-1)*width+(x+1)]) +
                 (-1* f[(y+1)*width + (x-1)]) + (-2*f[(y+1)*width+x]) + (-1*f[(y+1)*width+(x+1)]);
            output.f[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
        }
}

enum ACTIVATION {
	EQUAL,
	RELU,
	SIGMOID
};

const int BLOCK_SIZE = 4; // block size = 8 or 4
const int BLOCK_C = 4;

// 3*3 convolution, stride=1, padding=same, preserve dimensions
__global__ void convolve(const feature input, feature output, convlayer conv, int activation) {
	int ih = input.h;
	int iw = input.w;
	int ic = input.c;
	int oc = conv.n;

	__shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE][BLOCK_C];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tix = threadIdx.x;
    int tiy = threadIdx.y;
    int tiz = threadIdx.z;
    int layer = threadIdx.z + blockIdx.z * blockDim.z;
    int block_z = blockDim.z;
    if (0 <= x && x < iw && 0 <= y && y < ih) {
    	// some threads may be out of bounds
    	if (layer < oc) {
//    	for (int layer = 0; layer < oc; layer++) { // output layer
			float sum = conv.b[layer];
			for (int k = 0; k < ic; k++) { // input layer
				//if (threadIdx.z == 0) {
				if (k % block_z == 0) {
					if (k+tiz < ic) {
						sdata[tiy][tix][tiz] = input.f[y*iw*ic + x*ic + (k+tiz)];
					}
					__syncthreads();
				}

				for (int dx = -1; dx <= 1; dx++) {
					for (int dy = -1; dy <= 1; dy++) {
						int sx = x + dx;
						int sy = y + dy;
						if (sx >= 0 && sx < iw && sy >= 0 && sy < ih) {
							// input shape (h w c)
							// kernel shape (3 3 input_c output_c)
							if (0 <= tix+dx && tix+dx < BLOCK_SIZE &&
								0 <= tiy+dy && tiy+dy < BLOCK_SIZE) {

								sum += sdata[tiy + dy][tix + dx][k % block_z] *
									   conv.w[(dy+1)*3*ic*oc + (dx+1)*ic*oc + k*oc + layer];
							} else {
								sum += input.f[sy*iw*ic + sx*ic + k] *
								   conv.w[(dy+1)*3*ic*oc + (dx+1)*ic*oc + k*oc + layer];
							}
						}
					}
				}
			}
			// activation function
			if (activation == RELU) {
				if (sum < 0) {
					sum = 0;
				}
			} else if (activation == SIGMOID) {
				sum = 1 / (1 + exp(-sum));
			}
			output.f[layer + x*oc + y*iw*oc] = sum;
    	}
    }

}

// separate activation layer
__global__ void activation_gpu(const feature input, int activation) {
	int ih = input.h;
	int iw = input.w;
	int ic = input.c;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int layer = threadIdx.z + blockIdx.z * blockDim.z;
    if (0 <= x && x < iw && 0 <= y && y < ih) {
    	// some threads may be out of bounds
		for (int k = 0; k < ic; k++) { // layer
			float v = input.f[y*iw*ic + x*ic + k];
			if (activation == RELU) {
				if (v < 0) {
					v = 0;
				}
			} else if (activation == SIGMOID) {
				v = 1 / (1 + exp(-v));
			}
			input.f[y*iw*ic + x*ic + k] = v;
		}

    }
}

// dimensions (h/2, w/2)
__global__ void maxpool(const feature input, feature output) {
	int inh = input.h;
	int inw = input.w;
	int inc = input.c;
	int outh = inh / 2;
	int outw = inw / 2;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int layer = threadIdx.z + blockIdx.z * blockDim.z;
    if (0 <= x && x < outw && 0 <= y && y < outh) {
    	// some threads may be out of bounds
    	if (layer < inc) {
//    	for (int layer = 0; layer < inc; layer++) { // input and output layer
    		int bx = x * 2;
    		int by = y * 2;
    		float m1 = max(input.f[layer + bx*inc + by*inw*inc],
    					   input.f[layer + (bx+1)*inc + by*inw*inc]);
			float m2 = max(input.f[layer + bx*inc + (by+1)*inw*inc],
						   input.f[layer + (bx+1)*inc + (by+1)*inw*inc]);
			m1 = max(m1, m2);
			output.f[layer + x*inc + y*outw*inc] = m1;
    	}
    }
}

// 3*3 transposed convolution, stride=2, padding=same, dimensions (h*2, w*2)
__global__ void convolve_transpose(const feature input, feature output, convlayer conv, int activation) {
	int ih = input.h;
	int iw = input.w;
	int ic = input.c;
	int oh = (ih - 1) * 2 + 3 - 1; // (h-1)*stride + kernel - 2*pad + out_pad
	int ow = (iw - 1) * 2 + 3 - 1;
	int oc = conv.n;

	__shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE][BLOCK_C];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int layer = threadIdx.z + blockIdx.z * blockDim.z;
	int tix = threadIdx.x;
	int tiy = threadIdx.y;
	int tiz = threadIdx.z;
	int block_z = blockDim.z;
    if (0 <= x && x < ow && 0 <= y && y < oh) {
    	// some threads may be out of bounds
    	if (layer < oc) {
//    	for (int layer = 0; layer < oc; layer++) { // output layer
			float sum = conv.b[layer];
			for (int k = 0; k < ic; k++) { // input layer
				if (k % block_z == 0) {
					if (k+tiz < ic && y % 2 == 1 && x % 2 == 1) {
						sdata[tiy][tix][tiz] = input.f[(y/2)*iw*ic + (x/2)*ic + (k+tiz)];
					}
					__syncthreads();
				}
				for (int dx = -1; dx <= 1; dx++) {
					for (int dy = -1; dy <= 1; dy++) {
						int sx = x + dx;
						int sy = y + dy;
						if (0 <= sx && sx < ow && sx % 2 == 1 &&
								0 <= sy && sy < oh && sy % 2 == 1) {
							int insx = sx / 2;
							int insy = sy / 2;
							// input shape = (h w c)
							// kernel shape = (w h output_c input_c) !!!
							if (0 <= tix+dx && tix+dx < BLOCK_SIZE &&
								0 <= tiy+dy && tiy+dy < BLOCK_SIZE) {

								sum += sdata[tiy + dy][tix + dx][k % block_z] *
										conv.w[( 2-(dy+1) )*3*ic*oc + ( 2-(dx+1) )*ic*oc + layer*oc + k];
							} else {
								sum += input.f[insy*iw*ic + insx*ic + k] *
								   conv.w[( 2-(dy+1) )*3*ic*oc + ( 2-(dx+1) )*ic*oc + layer*oc + k];
							}
						}
					}
				}
			}
			// activation function
			if (activation == RELU) {
				if (sum < 0) {
					sum = 0;
				}
			} else if (activation == SIGMOID) {
				sum = 1 / (1 + exp(-sum));
			}
			output.f[layer + x*oc + y*ow*oc] = sum;
    	}
    }
}

// Useless general kernel function, can be deleted
__global__ void convolve_gpu(float *input, float *cpu, float *kernel, int height, int width, int k_size) {
	// x and y at center of convolution matrix
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// convolution without padding
	int k_half = k_size / 2;
	int min_y = k_half;
	int max_y = height - 1 - k_half;
	int min_x = k_half;
	int max_x = width - 1 - k_half;

	// some threads may be out of image boundaries
	if (min_y <= y && y <= max_y &&
		min_x <= x && x <= max_x)
	{
		float sum = 0;
		for (int ky = -k_half; ky <= k_half; ky++) {
			for (int kx = -k_half; kx <= k_half; kx++) {
				int img_y = y + ky;
				int img_x = x + kx;
				int pixel_idx = img_y * width + img_x;
				sum += input[pixel_idx] * kernel[(ky+k_half) * k_size + (kx+k_half)];
			}
		}
		cpu[y * width + x] = sum; // DEBUG: input[pixel]
	}
}


std::ifstream modelfile;
int read_w = 0;

convlayer read_layer(int prev_c, unsigned int c) {
	float *w = (float *)malloc(9*prev_c*c * sizeof(float));
	float *b = (float *)malloc(c * sizeof(float));
	// Read weights with shape (H, W, input, output)
	for (int i = 0; i < 9*prev_c*c; i++) {
		modelfile >> w[i];
		read_w++;
	}
	// Read bias with shape (output)
	for (int i = 0; i < c; i++) {
		modelfile >> b[i];
		read_w++;
	}
	convlayer layer = {
		alloc_cuda_copy_host_f(w, 9*prev_c*c),
		alloc_cuda_copy_host_f(b, c),
		c
	};
	free(w);
	free(b);
	return layer;
}

void scale_image(imgDataF img, float scale, int n) {
	for (int i = 0; i < n; i++) {
		img.pixels[i] *= scale;
	}
}

void image_info(imgDataF img, int img_size) {
	float min = img.pixels[0];
	float max = img.pixels[0];
	for (int i = 0; i < img_size; i++) {
		if (img.pixels[i] < min) min = img.pixels[i];
		if (img.pixels[i] > max) max = img.pixels[i];
	}
	printf("min: %f, max: %f\n", min, max);
}

int main(int argc, char **argv) {
	// Arg parsing

	int times_to_execute = 1;

	const char* short_options = "hb:m:";
	const struct option long_options[] = {
		{ "help", no_argument, NULL, 'h' },
		{ "benchmark", required_argument, NULL, 'b' },
		{ "model", required_argument, NULL, 'm' },
		{ NULL, 0, NULL, 0}
	};

	int res;
	int option_index;

	while (1) {
		res = getopt_long(argc, argv, short_options, long_options, NULL);
		if (res == -1) {
			break;
		} else if(res == 'h') {
			printf("Usage:\n"
					"-h, --help: print help\n"
					"-b, --benchmark <X>: print run time of NN averaged by X runs\n"
					"-m, --model <file.txt>: load NN weights from file\n");
			exit(0);
		} else if (res == 'b') {
			times_to_execute = atoi(optarg);
		} else if (res == 'm') {
			modelfile.open(optarg);
		}
	}

	if (optind >= argc) {
		printf("Expected image filename in args\n");
		exit(1);
	}

	// Load image and model weights

	printf("Image path: \"%s\"\n", argv[optind]);
	imgData img = loadImage(argv[optind]); // shape (h w 1), values in [0-255]
	printf("Loaded image\n");
	int img_size = img.height * img.width;

	if (!modelfile.is_open()) {
		modelfile.open("model2.txt");
	}

	// Allocate memory on CPU and GPU
	// Copy from CPU to GPU

	imgDataF imgf = {new float[img_size], img.width, img.height};
	byteToFloat(img, imgf, img_size);
	scale_image(imgf, 1.0/255, img_size);
	// image_info(imgf, img_size);
	feature img_tensor = {alloc_cuda_copy_host_f(imgf.pixels, img_size), img.height, img.width, 1};
	feature res_tensor = {alloc_cuda_zero_f(img_size), img.height, img.width, 1};

	feature c1 = {alloc_cuda_zero_f(img_size/1*32), img.height, img.width, 32};
	feature m1 = {alloc_cuda_zero_f(img_size/4*32), img.height/2, img.width/2, 32};
	feature c2 = {alloc_cuda_zero_f(img_size/4*32), img.height/2, img.width/2, 32};
	feature m2 = {alloc_cuda_zero_f(img_size/16*32), img.height/4, img.width/4, 32};
	feature t1 = {alloc_cuda_zero_f(img_size/4*32), img.height/2, img.width/2, 32};
	feature t2 = {alloc_cuda_zero_f(img_size*32), img.height, img.width, 32};

	convlayer cc1 = read_layer(1, 32);
	convlayer cc2 = read_layer(32, 32);
	convlayer ct1 = read_layer(32, 32);
	convlayer ct2 = read_layer(32, 32);
	convlayer cc3 = read_layer(32, 1);

	printf("Read %d weights\n", read_w);

	// Execute NN

	printf("Process image\n");
	auto t_start = std::chrono::system_clock::now();

	for (int runi = 0; runi < times_to_execute; runi++) {

		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_C);
		dim3 numBlocks1((img.width+BLOCK_SIZE-1)/BLOCK_SIZE,
				(img.height+BLOCK_SIZE-1)/BLOCK_SIZE, 32/BLOCK_C);

		dim3 numBlocks2((img.width/2+BLOCK_SIZE-1)/BLOCK_SIZE,
				(img.height/2+BLOCK_SIZE-1)/BLOCK_SIZE, 32/BLOCK_C);

		dim3 numBlocks3((img.width/4+BLOCK_SIZE-1)/BLOCK_SIZE,
				(img.height/4+BLOCK_SIZE-1)/BLOCK_SIZE, 32/BLOCK_C);

		dim3 threadsPerBlock0(BLOCK_SIZE, BLOCK_SIZE, 1);
		dim3 numBlocks0((img.width+BLOCK_SIZE-1)/BLOCK_SIZE,
				(img.height+BLOCK_SIZE-1)/BLOCK_SIZE, 1);

		convolve<<<numBlocks1, threadsPerBlock>>>(img_tensor, c1, cc1, RELU);
		maxpool<<<numBlocks2, threadsPerBlock>>>(c1, m1);
		convolve<<<numBlocks2, threadsPerBlock>>>(m1, c2, cc2, RELU);
		maxpool<<<numBlocks3, threadsPerBlock>>>(c2, m2);
		convolve_transpose<<<numBlocks2, threadsPerBlock>>>(m2, t1, ct1, RELU);
		convolve_transpose<<<numBlocks1, threadsPerBlock>>>(t1, t2, ct2, RELU);
		convolve<<<numBlocks0, threadsPerBlock0>>>(t2, res_tensor, cc3, SIGMOID);

		// Experiment - launch excessive threads

//		dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
//		dim3 numBlocks((img.width+BLOCK_SIZE-1)/BLOCK_SIZE,
//					   (img.height+BLOCK_SIZE-1)/BLOCK_SIZE, 1);
//
//		convolve<<<numBlocks, threadsPerBlock>>>(img_tensor, c1, cc1, RELU);
//		maxpool<<<numBlocks, threadsPerBlock>>>(c1, m1);
//		convolve<<<numBlocks, threadsPerBlock>>>(m1, c2, cc2, RELU);
//		maxpool<<<numBlocks, threadsPerBlock>>>(c2, m2);
//		convolve_transpose<<<numBlocks, threadsPerBlock>>>(m2, t1, ct1, RELU);
//		convolve_transpose<<<numBlocks, threadsPerBlock>>>(t1, t2, ct2, RELU);
//		convolve<<<numBlocks, threadsPerBlock>>>(t2, res_tensor, cc3, SIGMOID);

		// Experiment - separate ReLU and sigmoid activation from layer

//		convolve<<<numBlocks, threadsPerBlock>>>(img_tensor, c1, cc1, EQUAL);
//		activation_gpu<<<numBlocks, threadsPerBlock>>>(c1, RELU);
//		maxpool<<<numBlocks, threadsPerBlock>>>(c1, m1);
//		convolve<<<numBlocks, threadsPerBlock>>>(m1, c2, cc2, EQUAL);
//		activation_gpu<<<numBlocks, threadsPerBlock>>>(c2, RELU);
//		maxpool<<<numBlocks, threadsPerBlock>>>(c2, m2);
//		convolve_transpose<<<numBlocks, threadsPerBlock>>>(m2, t1, ct1, EQUAL);
//		activation_gpu<<<numBlocks, threadsPerBlock>>>(t1, RELU);
//		convolve_transpose<<<numBlocks, threadsPerBlock>>>(t1, t2, ct2, EQUAL);
//		activation_gpu<<<numBlocks, threadsPerBlock>>>(t2, RELU);
//		convolve<<<numBlocks, threadsPerBlock>>>(t2, res_tensor, cc3, EQUAL);
//		activation_gpu<<<numBlocks, threadsPerBlock>>>(res_tensor, SIGMOID);

		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // waits for completion

	}

	std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - t_start;
	printf("Average time on GPU: %.2lf milliseconds\n", time_gpu / times_to_execute * 1000);
	printf("Finished\n");

	// Copy from GPU to CPU, save to disk

	int out_size = img_size;
	imgDataF write_imgf(new float[out_size], img.width, img.height);
	CUDA_CHECK_RETURN(
			cudaMemcpy(write_imgf.pixels, res_tensor.f, (out_size*sizeof(float)), cudaMemcpyDeviceToHost) );
	scale_image(write_imgf, 255, out_size);
	imgData write_img(new byte[out_size], img.width, img.height);
	floatToByte(write_imgf, write_img, out_size);
	writeImage(argv[optind], "denoised", write_img);

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

