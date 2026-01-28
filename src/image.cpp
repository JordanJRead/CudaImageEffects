#include <string>
#include <string_view>
#include "../include/image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

Image::Image(std::string_view file) {
    int externalChannelCount;
    mData = stbi_load(file.data(), &mWidth, &mHeight, &externalChannelCount, STBI_rgb);
    mChannelCount = 3;

    if (!mData) {
        printf("Could not load file %s for reason: %s\n", file.data(), stbi_failure_reason());
        mReadError = true;
    }
}

Image::~Image() {
    stbi_image_free(mData);
}