#ifndef VEC_H
#define VEC_H

#include <cuda/std/array>
#include <type_traits>

typedef unsigned char byte;

template <typename T>
concept IsFloat = std::is_same_v<T, float>;

template <typename T>
concept IsByte = std::is_same_v<T, byte>;

template <size_t ChannelCount, typename T>
concept IsRGBFloat = std::is_same_v<T, float> && ChannelCount == 3;

template <size_t ChannelCount, typename T>
class Pixel {
public:
    __host__ __device__
    Pixel() {}

    __host__ __device__
    Pixel(const Pixel<ChannelCount, byte>& bytePixel) requires IsFloat<T> {
        for (size_t i{ 0 }; i < ChannelCount; ++i) {
            mData[i] = bytePixel[i] / 255.0f;
        }
    }
    
    __host__ __device__
    Pixel(const Pixel<ChannelCount, float>& floatPixel) requires IsByte<T> {
        for (size_t i{ 0 }; i < ChannelCount; ++i) {
            mData[i] = (byte)(floatPixel[i] * 255);
        }
    }
    
    // Returns the brightness of a 3-float pixel
    __host__ __device__
    float getBrightness() const requires IsRGBFloat<ChannelCount, T> {
        return 0.2126f * mData[0] + 0.7152f * mData[1] + 0.0722f * mData[2];
    }

    __host__ __device__
    T& operator[](size_t i) {
        if (i >= ChannelCount)
            return mData[0];
        return mData[i];
    }

    __host__ __device__
    T operator[](size_t i) const {
        return mData[i];
    }

private:
    cuda::std::array<T, ChannelCount> mData;
};

#endif