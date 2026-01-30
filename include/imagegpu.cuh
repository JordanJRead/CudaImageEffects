#ifndef IMAGE_GPU_H
#define IMAGE_GPU_H

#include <type_traits>
#include "imagecpu.h"
#include "pixel.cuh"
#include <stdio.h>

typedef unsigned char byte;
typedef cuda::std::pair<size_t, size_t> Indices;

template <size_t ChannelCount, typename T>
concept IsRGB8 = ChannelCount == 3 && std::is_same_v<T, byte>;

template <size_t ChannelCount, typename T>
class ImageGPU {
public:

    static ImageGPU copyFromCPU(const ImageCPU& cpuImage) requires IsRGB8<ChannelCount, T> {
        ImageGPU<ChannelCount, T> gpuImage;
        gpuImage.mWidth = cpuImage.getWidth();
        gpuImage.mHeight = cpuImage.getHeight();

        if (cudaMalloc(&gpuImage.mData, gpuImage.getByteCount()) != cudaSuccess) {
            printf("GPU image allocation failed");
        }
        if (cudaMemcpy(gpuImage.mData, cpuImage.getData(), cpuImage.getByteCount(), cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("GPU image copying failed");
        }
        return gpuImage;
    }

    static ImageGPU copyDimFromCPU(const ImageCPU& cpuImage) {
        ImageGPU<ChannelCount, T> gpuImage;
        gpuImage.mWidth = cpuImage.getWidth();
        gpuImage.mHeight = cpuImage.getHeight();

        if (cudaMalloc(&gpuImage.mData, gpuImage.getByteCount()) != cudaSuccess) {
            printf("GPU image allocation failed");
        }
        return gpuImage;
    }

    __host__ __device__
    ~ImageGPU() {
        if (mIsOwner) {
            cudaFree(mData);
        }
    }

    __device__
    ImageGPU(const ImageGPU& other)
        : mWidth{ other.mWidth }
        , mHeight{ other.mHeight }
        , mData{ other.mData }
        , mIsOwner{ false }
    {}

    ImageGPU(ImageGPU<ChannelCount, T>&& other) {
        mWidth = other.mWidth;
        mHeight = other.mHeight;
        mData = other.mData;
        mIsOwner = other.mIsOwner;
        other.mIsOwner = false;
    }

    ImageGPU& operator=(const ImageGPU&) = delete;
    ImageGPU& operator=(ImageGPU&&) = delete;
    
    __device__
    Indices getPixelIndices(dim3 blockDim, dim3 blockIdx, dim3 threadIdx) const {
        Indices indices{ 
            blockDim.x * blockIdx.x + threadIdx.x,
            blockDim.y * blockIdx.y + threadIdx.y
        };
        if (indices.first >= mWidth || indices.second >= mHeight)
            return { (size_t)-1, (size_t)-1 };
        return indices;
    }

    __device__
    Pixel<ChannelCount, T> sample(const Indices& indices) const {
        int dataIndex = (indices.first + indices.second * mWidth) * ChannelCount;

        Pixel<ChannelCount, T> pixel;

        for (size_t i{ 0 }; i < ChannelCount; ++i) {
            pixel[i] = mData[dataIndex + i];
        }

        return pixel;
    }

    __device__
    void setPixel(const Indices& indices, const Pixel<ChannelCount, T>& pixel) {
        int dataIndex = (indices.first + indices.second * mWidth) * ChannelCount;

        for (size_t i{ 0 }; i < ChannelCount; ++i) {
            mData[dataIndex + i] = pixel[i];
        }
    }

    __device__ __host__
    int getWidth() const { return mWidth; }
    
    __device__ __host__
    int getHeight() const { return mHeight; }

    __host__
    const T* getData() const { return mData; }

    int getByteCount() {
        return mWidth * mHeight * ChannelCount * sizeof(T);
    }

private:
    T* mData;
    int mWidth;
    int mHeight;
    bool mIsOwner{ true };
    ImageGPU() = default;
};

#endif