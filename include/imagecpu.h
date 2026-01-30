#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <string_view>

typedef unsigned char byte;

template <size_t ChannelCount, typename T>
class ImageGPU;

class ImageCPU {
public:
    ImageCPU(std::string_view file);
    ImageCPU(const ImageGPU<3, byte>& gpuImage);

    ImageCPU(const ImageCPU&) = delete;
    ImageCPU& operator=(const ImageCPU&) = delete;
    ImageCPU& operator=(ImageCPU&&) = delete;
    ImageCPU(ImageCPU&& other) {
        mWidth = other.mWidth;
        mHeight = other.mHeight;
        mChannelCount = other.mChannelCount;
        mData = other.mData;
        other.mData = nullptr;
    }

    ~ImageCPU();
    void writeToFile(std::string_view file);

    int getWidth() const { return mWidth; }
    int getHeight() const { return mHeight; }
    int getChannelCount() const { return mChannelCount; }
    int getByteCount() const { return mWidth * mHeight * mChannelCount; }
    bool getReadError() const { return mReadError; }
    bool getWriteError() const { return mWriteError; }
    const byte* getData() const { return mData; }

private:
    bool mReadError{ false };
    bool mWriteError{ false };
    byte* mData;
    int mWidth;
    int mHeight;
    int mChannelCount;
};

#endif