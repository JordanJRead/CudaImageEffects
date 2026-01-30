#ifndef SORT_H
#define SORT_H

#include "imagegpu.cuh"
#include "imagecpu.h"

namespace Sort {

    namespace {   
        __global__
        void KERNALCreateValue(const ImageGPU<3, byte> sourceImage, ImageGPU<1, float> valueImage) {
            Indices indices = sourceImage.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;

            Pixel<3, float> pixel{ sourceImage.sample(indices) };

            float brightness = pixel.getBrightness();
            Pixel<1, float> valuePixel;
            valuePixel[0] = brightness;

            valueImage.setPixel(indices, valuePixel);
        }

        __global__
        void KERNALCreateValueDisplay(const ImageGPU<1, float> valueImage, ImageGPU<3, byte> displayImage) {
            Indices indices = valueImage.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;

            Pixel<1, float> pixel{ valueImage.sample(indices) };

            Pixel<3, float> displayPixel;
            displayPixel[0] = pixel[0];
            displayPixel[1] = pixel[0];
            displayPixel[2] = pixel[0];

            displayImage.setPixel(indices, displayPixel);
        }
    }

    ImageCPU sortImage(const ImageCPU& sourceImage) {
        constexpr int BLOCK_WIDTH{ 16 };
        dim3 blockDim = dim3(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 blockCount = dim3((sourceImage.getWidth() + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (sourceImage.getHeight() + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

        ImageGPU<1, float> gpuValueImage{ ImageGPU<1, float>::copyDimFromCPU(sourceImage) };
        ImageGPU<3, byte> gpuSourceImage{ ImageGPU<3, byte>::copyFromCPU(sourceImage) };

        KERNALCreateValue<<<blockCount, blockDim>>>(gpuSourceImage, gpuValueImage);

        ImageGPU<3, byte> gpuDisplayValueImage{ ImageGPU<3, byte>::copyDimFromCPU(sourceImage) };

        KERNALCreateValueDisplay<<<blockCount, blockDim>>>(gpuValueImage, gpuDisplayValueImage);

        ImageCPU displayImage{ gpuDisplayValueImage };
        return displayImage;
    }
}

#endif