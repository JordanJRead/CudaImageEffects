#ifndef VEC_H
#define VEC_H

#include <cuda/std/array>
#include <type_traits>

typedef unsigned char byte;

template <typename T>
concept IsFloating = std::is_same_v<T, float> || std::is_same_v<T, __half>;

template <typename T>
concept IsByte = std::is_same_v<T, byte>;

template <size_t ChannelCount, typename T>
concept IsRGBFloating = IsFloating<T> && ChannelCount == 3;

template <size_t ChannelCount, typename T>
class Pixel {
public:
    __host__ __device__
    Pixel() {}
    Pixel(const Pixel<ChannelCount, T>&) = default;

    __host__ __device__
    Pixel(const Pixel<ChannelCount, byte>& bytePixel) requires IsFloating<T> {
        for (size_t i{ 0 }; i < ChannelCount; ++i) {
            mData[i] = (T)(bytePixel[i] / 255.0f);
        }
    }
    
    template <typename FloatingType>
    __host__ __device__
    Pixel(const Pixel<ChannelCount, FloatingType>& floatPixel) requires (IsByte<T> && IsFloating<FloatingType>) {
        for (size_t i{ 0 }; i < ChannelCount; ++i) {
            mData[i] = (byte)(floatPixel[i] * 255);
        }
    }

    __host__ __device__
    Pixel(const Pixel<ChannelCount, int16_t>& intPixel) requires IsByte<T> {
        for (size_t i{ 0 }; i < ChannelCount; ++i) {
            mData[i] = intPixel[0] > 0 ? 255 : 0;
        }
    }
    
    // Returns the brightness of a 3-float pixel
    __host__ __device__
    float getBrightness() const requires IsRGBFloating<ChannelCount, T> {
        return (T)0.2126f * mData[0] + (T)0.7152f * mData[1] + (T)0.0722f * mData[2];
    }

    __host__ __device__
    T& operator[](size_t i) {
        return mData[i];
    }

    __host__ __device__
    const T& operator[](size_t i) const {
        return mData[i];
    }

private:
    cuda::std::array<T, ChannelCount> mData = {};
};

#endif