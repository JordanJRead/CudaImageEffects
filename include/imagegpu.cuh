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
    __host__
    ImageGPU(const ImageCPU& cpuImage) requires IsRGB8<ChannelCount, T> {
        mWidth = cpuImage.getWidth();
        mHeight = cpuImage.getHeight();

        if (cudaMalloc(&mData, cpuImage.getByteCount()) != cudaSuccess) {
            printf("GPU image allocation failed");
        }
        if (cudaMemcpy(mData, cpuImage.getData(), cpuImage.getByteCount(), cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("GPU image copying failed");
        }
    }

    __host__ __device__
    ~ImageGPU() {
        if (!mIsCopy) {
            cudaFree(mData);
        }
    }

    __device__
    ImageGPU(const ImageGPU& other)
        : mWidth{ other.mWidth }
        , mHeight{ other.mHeight }
        , mData{ other.mData }
        , mIsCopy{ true }
    {}

    ImageGPU(ImageGPU&&) = delete;
    ImageGPU& operator=(const ImageGPU&) = delete;
    ImageGPU& operator=(ImageGPU&&) = delete;
    
    __device__
    Indices getPixelIndices(dim3 blockDim, dim3 blockIdx, dim3 threadIdx) {
        Indices indices{ 
            blockDim.x * blockIdx.x + threadIdx.x,
            blockDim.y * blockIdx.y + threadIdx.y
        };
        if (indices.first >= mWidth || indices.second >= mHeight)
            return { (size_t)-1, (size_t)-1 };
        return indices;
    }

    __device__
    Pixel<ChannelCount, T> sample(const Indices& indices) {
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

private:
    T* mData;
    int mWidth;
    int mHeight;
    bool mIsCopy{ false };
};

#endif