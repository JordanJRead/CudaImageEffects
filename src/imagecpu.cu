#include <string>
#include <string_view>
#include "../include/imagecpu.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/imagecpu.h"
#include "../include/imagegpu.cuh"

ImageCPU::ImageCPU(std::string_view file) {
    int externalChannelCount;
    mData = stbi_load(file.data(), &mWidth, &mHeight, &externalChannelCount, STBI_rgb);
    mChannelCount = 3;

    if (!mData) {
        printf("Could not load file %s for reason: %s\n", file.data(), stbi_failure_reason());
        mReadError = true;
    }
}

ImageCPU::ImageCPU(const ImageGPU<3, byte>& gpuImage) {
    mWidth = gpuImage.getWidth();
    mHeight = gpuImage.getHeight();
    mChannelCount = 3;
    mData = (byte*)malloc(getByteCount());
    cudaMemcpy(mData, gpuImage.getData(), getByteCount(), cudaMemcpyDeviceToHost);
}

void ImageCPU::writeToFile(std::string_view file) {
    int returnCode = stbi_write_png(file.data(), mWidth, mHeight, mChannelCount, mData, mWidth * 3);
    if (returnCode == 0) {
        mWriteError = true;
        printf("Could not write to file %s for reason: %s\n", file.data(), stbi_failure_reason());
    }
}

ImageCPU::~ImageCPU() {
    stbi_image_free(mData);
}