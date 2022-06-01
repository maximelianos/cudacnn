# CUDA CNN

Simple neural network convolutional encoder-decoder written in C++/CUDA.
Model is first trained in Python Keras, weights are copied in simple text format
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
* `model2.txt` - all models weights in text format
* `desc2.txt` - model architecture


# References

https://github.com/lukas783/CUDA-Sobel-Filter
https://github.com/ULHPC/tutorials/blob/devel/cuda/exercises/convolution/LoG_gpu_exercise.cu
https://keras.io/examples/vision/autoencoder/
https://github.com/lvandeve/lodepng
