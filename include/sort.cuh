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

        // Only works for float and byte
        template <typename T>
        __global__
        void KERNALOneValueToRBG8(const ImageGPU<1, T> oneChannelImage, ImageGPU<3, byte> rgb8Image) {
            Indices indices = oneChannelImage.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;

            Pixel<1, T> pixel{ oneChannelImage.sample(indices) };

            Pixel<3, T> displayPixel;
            displayPixel[0] = pixel[0];
            displayPixel[1] = pixel[0];
            displayPixel[2] = pixel[0];

            rgb8Image.setPixel(indices, displayPixel);
        }

        __global__
        void KERNALCreateContrastMask(const ImageGPU<3, byte> sourceImage, ImageGPU<1, byte> contrastMask, float minBrightness, float maxBrightness) {
            Indices indices = sourceImage.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;
            
            float brightness = Pixel<3, float>{ sourceImage.sample(indices) }.getBrightness();
            bool mask = brightness > minBrightness && brightness < maxBrightness;
            Pixel<1, byte> maskPixel;
            maskPixel[0] = mask ? 255 : 0;
            contrastMask.setPixel(indices, maskPixel);
        }

        __global__
        void KERNALCreateStrideMask(const ImageGPU<1, byte> contrastMask, ImageGPU<1, int16_t> strideMask) {
            Indices indices = contrastMask.getPixelIndices(blockDim, blockIdx, threadIdx, true);
            if (indices.first == (size_t)-1)
                return;

            bool isInStride{ false };
            int16_t strideLength{ 0 };
            Indices strideStartIndices;

            int height = contrastMask.getHeight();
            for (int i{ 0 }; i < 4; ++i) {
                Pixel<1, int16_t> pixel;
                pixel[0] = 1;
                strideMask.setPixel(strideStartIndices, pixel);
                break;
                indices.second += 1;
                
                // bool shouldThisPixelBeBlack{ true };
                // bool passMask{ contrastMask.sample(indices)[0] != 0 };
                // if (passMask) {
                //     strideLength++;

                //     // Start a stride
                //     if (!isInStride) {
                //         strideStartIndices = indices;
                //         isInStride = true;
                //         shouldThisPixelBeBlack = false;
                //     }
                // }
                // else {
                //     // Exit stride
                //     if (isInStride) {
                //         isInStride = false;
                //         //Pixel<1, int16_t> pixel;
                //         //pixel[0] = strideLength;
                //         //strideMask.setPixel(strideStartIndices, pixel);
                //         strideLength = 0;
                //     }
                // }

                // if (shouldThisPixelBeBlack) {
                //     // Pixel<1, int16_t> pixel;
                //     // pixel[0] = 0;
                //     //strideMask.setPixel(strideStartIndices, pixel);
                // }

            }
        }

        __global__
        void KERNALCreateVisualFromStrideMask(const ImageGPU<1, int16_t> strideMask, ImageGPU<3, byte> displayImage) {
            Indices indices = strideMask.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;
            
            int16_t startingStrideLength{ 1 };
            int16_t strideLengthToGo{ 0 };
            
            int height = strideMask.getHeight();
            for (int i{ 0 }; i < height; ++i) {
                int16_t strideMaskValue{ strideMask.sample(indices)[0] };
                if (strideMaskValue > 0) {
                    startingStrideLength = strideMaskValue;
                    strideLengthToGo = strideMaskValue;
                }

                float progress = (float)strideLengthToGo / startingStrideLength;
                Pixel<3, float> pixel;
                pixel[0] = progress;
                pixel[1] = progress;
                pixel[2] = progress;
                displayImage.setPixel(indices, pixel);

                strideLengthToGo--;
                if (strideLengthToGo < 0)
                    strideLengthToGo = 0;
            
                indices.second += 1;
            }
        }
    }

    ImageCPU sortImage(const ImageCPU& sourceImage) {
        constexpr int SQUARE_BLOCK_WIDTH{ 16 };
        dim3 squareThreadsPerBlock = dim3(SQUARE_BLOCK_WIDTH, SQUARE_BLOCK_WIDTH);
        dim3 squareBlockCount = dim3((sourceImage.getWidth() + SQUARE_BLOCK_WIDTH - 1) / SQUARE_BLOCK_WIDTH, (sourceImage.getHeight() + SQUARE_BLOCK_WIDTH - 1) / SQUARE_BLOCK_WIDTH);

        dim3 lineThreadsPerBlock = dim3(2);
        dim3 lineBlockCount = dim3((sourceImage.getWidth() + 1) / 2);
        printf("%u %u %u %u %u %u\n", lineThreadsPerBlock.x, lineThreadsPerBlock.y, lineThreadsPerBlock.z, lineBlockCount.x, lineBlockCount.y, lineBlockCount.z);

        ImageGPU<3, byte> gpuSourceImage{ ImageGPU<3, byte>::copyFromCPU(sourceImage) };
        ImageGPU<1, float> gpuValueImage{ ImageGPU<1, float>::copyDimFromCPU(sourceImage) };
        ImageGPU<1, byte> gpuContrastMaskImage{ ImageGPU<1, byte>::copyDimFromCPU(sourceImage) };
        ImageGPU<1, int16_t> gpuStrideMaskImage{ ImageGPU<1, int16_t>::copyDimFromCPU(sourceImage) };

        KERNALCreateValue<<<squareBlockCount, squareThreadsPerBlock>>>(gpuSourceImage, gpuValueImage);
        cudaDeviceSynchronize();
        KERNALCreateContrastMask<<<squareBlockCount, squareThreadsPerBlock>>>(gpuSourceImage, gpuContrastMaskImage, 0.3, 0.7);
        cudaDeviceSynchronize();
        KERNALCreateStrideMask<<<lineBlockCount, lineThreadsPerBlock>>>(gpuContrastMaskImage, gpuStrideMaskImage);
        cudaDeviceSynchronize();


        ImageGPU<3, byte> gpuDisplayValueImage{ ImageGPU<3, byte>::copyDimFromCPU(sourceImage) };
        //KERNALCreateVisualFromStrideMask<<<lineBlockCount, lineThreadsPerBlock>>>(gpuStrideMaskImage, gpuDisplayValueImage);
        KERNALOneValueToRBG8<<<squareBlockCount, squareThreadsPerBlock>>>(gpuStrideMaskImage, gpuDisplayValueImage);

        ImageCPU displayImage{ gpuDisplayValueImage };
        return displayImage;
    }
}

#endif