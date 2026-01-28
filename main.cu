#include "stdio.h"
#include "include/colour.cuh"
#include <string>
#include <string_view>
#include <array>
#include <filesystem>
#include "include/effect.cuh"
#include "include/image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#include "include/effecttype.h"

typedef unsigned char byte;

__global__
void doImageEffect(EffectType::Type effect, int width, int height, int channelCount, byte* data) {
    int pixelIndexX = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelIndexY = blockDim.y * blockIdx.y + threadIdx.y;
    int flatPixelIndex = pixelIndexX + width * pixelIndexY;
    int dataIndex = flatPixelIndex * channelCount;

    if (pixelIndexX >= width || pixelIndexY >= height)
        return;
    
    Colour originalColour{
        data[dataIndex + 0],
        data[dataIndex + 1],
        data[dataIndex + 2]
    };

    Colour output;
    switch (effect) {
        case EffectType::stipple:
            output = Effect::doStipple(originalColour, pixelIndexX, pixelIndexY);
            break;
        case EffectType::invert:
            output = Effect::doInvert(originalColour);
            break;
    }

    data[dataIndex + 0] = output.getByteR();
    data[dataIndex + 1] = output.getByteG();
    data[dataIndex + 2] = output.getByteB();
}

int main(int argc, char* argv[]) {

    // Input validation
    if (argc != 3 && argc != 4) {
        printf("ARGUMENT COUNT ERROR");
        return 1;
    }

    std::string effectTypeInput{ argv[1] };
    std::filesystem::path filePath{ argv[2] };
    std::filesystem::path outputFilePath;

    EffectType::Type effectType{ EffectType::getTypeFromString(effectTypeInput) };
    if (effectType == EffectType::max_effects) {
        printf("Usage: ./main.exe EFFECT IMAGEFILE [OUTPUTFILE].\nEFFECT is of: %s", EffectType::getPipedEffectNames().c_str());
        return 1;
    }

    // Output name
    if (argc == 4) {
        outputFilePath = argv[3];
    }
    else {
        std::string outputFilePathName{ filePath.parent_path().string().c_str() };
        outputFilePathName += "/";
        outputFilePathName += filePath.stem().string().c_str();
        outputFilePathName += "_";
        outputFilePathName += EffectType::effectNames[effectType];
        outputFilePathName += ".png";
        outputFilePath = outputFilePathName;
    }

    // Load image data
    Image image{ filePath.string().c_str() };
    if (image.getReadError()) {
        return 1;
    }
    int byteCount = image.getByteCount();

    // Data
    byte* d_data;
    cudaMalloc(&d_data, byteCount);
    cudaMemcpy(d_data, image.getData(), byteCount, cudaMemcpyHostToDevice);

    #define BLOCK_WIDTH 16

    dim3 blockDim = dim3(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 blockCount = dim3((image.getWidth() + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (image.getHeight() + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

    doImageEffect<<<blockCount, blockDim>>>(effectType, image.getWidth(), image.getHeight(), 3, d_data);

    byte* returnData = (byte*)malloc(byteCount);

    cudaMemcpy(returnData, d_data, byteCount, cudaMemcpyDeviceToHost);

    // fixme double stipple doesn't do anything?
    // fixme actually nothing works - git repo is made locally unstaged?
    // todo make way to write out image with class (ctor with data instead of file? pointer stuff though)
    int returnCode = stbi_write_png(outputFilePath.string().c_str(), image.getWidth(), image.getHeight(), image.getChannelCount(), returnData, image.getWidth() * 3);
    if (returnCode == 0) {
        printf("Could not write to file %s for reason: TODO ADD REASON\n", filePath.string().c_str());
    free(returnData);
    cudaFree(d_data); // todo make sure this gets deallocated (raii?)
    }

    free(returnData);
    cudaFree(d_data);
    return 0;
}