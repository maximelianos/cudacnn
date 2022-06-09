# CUDA CNN

Github: https://github.com/maximelianos/cudacnn

Simple neural network image denoiser written in C++/CUDA.
and encoder-decoder model is trained with Python Keras, weights are copied in simple text format
to be executed by C++.

# Build

Developed with API CUDA, OS Ubuntu 14.04 x86-64, cudatoolkit v8.0, gcc v4.8.4,
GPU NVIDIA GeForce GTX 770M, compute capability 3.0. Change compute capability of your GPU in Makefile.

```
make
./denoiser --benchmark <n> --model <model.txt> <image.png>
```

Options

* `--benchmark <n>` - run model n times on GPU, show average run time
* `--model <model.txt>` - load model weights from file
* `<image.png>` - image to be processed. Output image path is image_denoised.png

Remove generated files: `make distclean`

# Code structure

* `src/` - C++/CUDA program
* `autoencoder.ipynb` - Python notebook for training model in Keras ([original notebook](https://keras.io/examples/vision/autoencoder/))
* `model2.txt` - all model weights in text format
* `desc2.txt` - model architecture

# Speed comparison

Several optimizations were attempted:
* Join convolution and activation layer into one CUDA kernel (**conv+act**)
* Parallel execution of convolution channels in one layer; recompute grid size for each layer (**effective threads**)
* Reduce thread block size from 8 to 4 (**block=4**)
* Copy input layer to local shared memory (**shared mem**)

All convolutions have 3x3 range. Convolution weights of one layer have shape (3, 3, input_c, output_c) where input_c is number of channels in previous features, output_c is similar. The baseline implementation has separate convolution and activation kernels, threads are parallel by output feature height and width, thread grid is not adjusted between layers (which have different dimensions). Joint convolution and activation does not improve speed a lot. Recomputing the grid size and channel-wise parallelism removes most of non-functioning threads, and allows more threads to be executed at once. Small images have few kernels to execute, thus the speedup is larger. Reducing thread block size makes fewer threads that are out of image bounds. Shared memory is hard to implement, synchronization between threads may reduce performance.

<img src=runtime.png>

# References

1. https://github.com/lukas783/CUDA-Sobel-Filter
2. https://github.com/ULHPC/tutorials/blob/devel/cuda/exercises/convolution/LoG_gpu_exercise.cu
3. https://keras.io/examples/vision/autoencoder/
4. https://github.com/lvandeve/lodepng
