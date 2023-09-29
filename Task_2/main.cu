#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "IL/il.h"
#include <IL/devil_cpp_wrapper.hpp>
#include <stdlib.h>
#include <chrono>
#include <cassert>

using namespace std;

const int nChannels = 3;
const int threadsPerBlockSide = 16;


struct Image {
    ILubyte *channels[nChannels];
    ILubyte *channels_[nChannels];

    int width, height;
    int size;

    Image(ILubyte *image, int width, int height) {
        width = width;
        height = height;
        size = width * height;

        for (int c = 0; c < nChannels; c++) {
            assert(cudaSuccess == cudaMallocHost((void**)&channels[c], size*sizeof(ILubyte)));
            assert(cudaSuccess == cudaMallocHost((void**)&channels_[c], size*sizeof(ILubyte)));

            for (int i = 0; i < size; i++)
                channels[c][i] = image[3*i + c];
        }
    }

    Image(int width, int height) {
        width = width;
        height = height;
        size = width * height;

        for (int c = 0; c < nChannels; c++) {
            assert(cudaSuccess == cudaMallocHost((void**)&channels[c], size*sizeof(ILubyte)));
            assert(cudaSuccess == cudaMallocHost((void**)&channels_[c], size*sizeof(ILubyte)));
        }
    }

    ~Image() {
        for (int c = 0; c < nChannels; c++) {
            assert(cudaSuccess == cudaFreeHost(channels[c]));
            assert(cudaSuccess == cudaFreeHost(channels_[c]));
        }
    }

    void output(ILubyte *image_) {
        for (int c = 0; c < nChannels; c++)
            for (int i = 0; i < size; i++)
                image_[3*i + c] = channels_[c][i];
    }
};


__host__ __device__ void boxBlurFilterSequent(ILubyte *channel, ILubyte *channel_, int w, int h, int filter_radius, int x, int y) {
    ILfloat norm = (2 * filter_radius + 1) * (2 * filter_radius + 1);
    int output_value = 0;

    for(int i = x-filter_radius; i<=x+filter_radius; i++) {
        for(int j = y-filter_radius; j <= y+filter_radius; j++) {
            int i_ = abs(i) % h, j_ = abs(j) % w;

            if (i >= h) i_ = h - 1 - i_;
            if (j >= w) j_ = w - 1 - j_;

            output_value += channel[i_*w + j_];
        }
    }

    channel_[x*w + y] = (ILuint)((ILfloat)output_value / norm);
}


__global__ void boxBlurFilterParallel(ILubyte *channel, ILubyte *channel_, int w, int h, int filter_radius) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < h && y < w) boxBlurFilterSequent(channel, channel_, w, h, filter_radius, x, y);
}


int main(int argc, const char * argv[]) {
    float sequent_duration, parallel_duration_pure, parallel_duration;
    cudaEvent_t parallel_start_pure, parallel_stop_pure, parallel_start, parallel_stop;

    cudaStream_t streams[nChannels];
    for (int c = 0; c < nChannels; c ++)
        cudaStreamCreate(&streams[c]);

    if (argc != 2) {
        printf("Wrong arguments number: %d\n", argc - 1);
        return 1;
    }
    const int filterSize = stoi(argv[1]);

    ilInit();
    ilEnable(IL_ORIGIN_SET);

    ILuint handle;
    ilGenImages(1, &handle);
	ilBindImage(handle);
	ILboolean loaded = ilLoadImage("input.bmp");

    if (loaded == IL_FALSE) {
        printf("%s\n", iluErrorString(ilGetError()));
        return 1;
    }

    ILuint w = ilGetInteger(IL_IMAGE_WIDTH);
	ILuint h = ilGetInteger(IL_IMAGE_HEIGHT);
    int imageSize = w * h;

    int image_memory = imageSize * 3 * sizeof(ILubyte);
    ILubyte *h_input_image = new ILubyte[image_memory];
    ILubyte *h_output_image_ref = new ILubyte[image_memory];

    ilCopyPixels(0, 0, 0, w, h, 1, IL_RGB, IL_UNSIGNED_BYTE, h_input_image);

    Image *image_ref = new Image(h_input_image, w, h);

    // ===== CPU sequent =====
    chrono::steady_clock::time_point sequent_start = chrono::steady_clock::now();
    for (int x = 0; x < h; x++)
        for (int y = 0; y < w; y++)
            for (int c = 0; c < nChannels; c++)
                boxBlurFilterSequent(
                    image_ref -> channels[c],
                    image_ref -> channels_[c], 
                    w, h, filterSize, x, y
                );
    chrono::steady_clock::time_point sequent_stop = chrono::steady_clock::now();

    sequent_duration = (float)chrono::duration_cast<chrono::milliseconds>(sequent_stop - sequent_start).count();

    image_ref -> output(h_output_image_ref);
    // =======================
  
    // ===== GPU parallel =====
    ILubyte *h_output_image = new ILubyte[image_memory];  

    const int blocksPerGridX = (h + threadsPerBlockSide - 1) / threadsPerBlockSide;
    const int blocksPerGridY = (w + threadsPerBlockSide - 1) / threadsPerBlockSide;

    dim3 blockDim = dim3(threadsPerBlockSide, threadsPerBlockSide, 1);
    dim3 gridDim = dim3(blocksPerGridX, blocksPerGridY, 1);

    Image *h_image = new Image(h_input_image, w, h);

    ILubyte *d_channels[nChannels];
    ILubyte *d_channels_[nChannels];

    for (int c = 0; c < nChannels; c++) {
        assert(cudaSuccess == cudaMalloc((void**)&d_channels[c], sizeof(ILubyte)*imageSize));
        assert(cudaSuccess == cudaMalloc((void**)&d_channels_[c], sizeof(ILubyte)*imageSize));
    }

    assert(cudaSuccess == cudaEventCreate(&parallel_start));
    assert(cudaSuccess == cudaEventCreate(&parallel_stop));
    assert(cudaSuccess == cudaEventCreate(&parallel_start_pure));
    assert(cudaSuccess == cudaEventCreate(&parallel_stop_pure));

    assert(cudaSuccess == cudaEventRecord(parallel_start));

    for (int c = 0; c < nChannels; c++)
        assert(cudaSuccess == cudaMemcpyAsync(
            d_channels[c],
            h_image -> channels[c],
            sizeof(ILubyte)*imageSize,
            cudaMemcpyHostToDevice,
            streams[c]
        ));

    assert(cudaSuccess == cudaEventRecord(parallel_start_pure));

    for (int c = 0; c < nChannels; c++)
        boxBlurFilterParallel<<<gridDim, blockDim, 0, streams[c]>>>(d_channels[c], d_channels_[c], w, h, filterSize);

    assert(cudaSuccess == cudaEventRecord(parallel_stop_pure));
    assert(cudaSuccess == cudaEventSynchronize(parallel_stop_pure));
    assert(cudaSuccess == cudaEventElapsedTime(&parallel_duration_pure, parallel_start_pure, parallel_stop_pure));

    for (int c = 0; c < nChannels; c++)
        assert(cudaSuccess == cudaMemcpyAsync(
            h_image -> channels_[c],
            d_channels_[c],
            sizeof(ILubyte)*imageSize,
            cudaMemcpyDeviceToHost,
            streams[c]
        ));

    assert(cudaSuccess == cudaEventRecord(parallel_stop));
    assert(cudaSuccess == cudaEventSynchronize(parallel_stop));
    assert(cudaSuccess == cudaEventElapsedTime(&parallel_duration, parallel_start, parallel_stop));

    h_image -> output(h_output_image);

    // ========================

    ilSetPixels(0, 0, 0, w, h, 1, IL_RGB, IL_UNSIGNED_BYTE, h_output_image);
    ILboolean saved = ilSaveImage("result.bmp");

    if (saved == IL_FALSE) {
        printf("%s\n", iluErrorString(ilGetError()));
        return 1;
    }

    ilDeleteImages(1, &handle);

    bool is_correct = true;
    for (int i = 0; i < imageSize * 3; i++)
        if (h_output_image[i] != h_output_image_ref[i]) {
            printf("Status: \x1B[31mWrong:\033[0m\n");
            is_correct = false;
            break;
        }
    if (is_correct) {
        printf("Status: \x1B[32mCorrect\033[0m\n");
        
        printf("Elapsed speed (Sequent CPU) = %.2f MPx/s\n", 0.001 * imageSize / sequent_duration);
        printf("Elapsed speed (Parallel GPU pure) = %.2f MPx/s\n", 0.001 * imageSize / parallel_duration_pure);
        printf("Elapsed speed (Parallel GPU) = %.2f MPx/s\n", 0.001 * imageSize / parallel_duration);

        printf("Pure acceleration: %.2f\n", sequent_duration/parallel_duration_pure);
        printf("Acceleration: %.2f\n", sequent_duration/parallel_duration);
    }

    delete [] h_input_image;
    delete [] h_output_image;
    delete [] h_output_image_ref;
    delete image_ref;
    delete h_image;

    for (int c = 0; c < nChannels; c++) {
        assert(cudaSuccess == cudaFree(d_channels[c]));
        assert(cudaSuccess == cudaFree(d_channels_[c]));
    }

    for (int c = 0; c < nChannels; c ++)
        assert(cudaSuccess == cudaStreamDestroy(streams[c]));

    cout << cudaGetErrorString(cudaGetLastError()) << endl;

    return 0;
}
