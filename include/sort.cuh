#ifndef SORT_H
#define SORT_H

#include "imagegpu.cuh"
#include "imagecpu.h"
#include "timer.h"

namespace Sort {

    enum class Direction {
        horizontal,
        vertical
    };

    namespace {   
        __global__
        void KERNELRGB8ToValue(const ImageGPU<3, byte> sourceImage, ImageGPU<1, __half> valueImage) {
            Indices indices = sourceImage.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;

            Pixel<3, __half> pixel{ sourceImage.sample(indices) };

            __half brightness = pixel.getBrightness();
            Pixel<1, __half> valuePixel;
            valuePixel[0] = brightness;

            valueImage.setPixel(indices, valuePixel);
        }

        // Only works for float and byte
        template <typename T>
        __global__
        void KERNELOneValueToRGB8(const ImageGPU<1, T> oneChannelImage, ImageGPU<3, byte> rgb8Image) {
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
        void KERNELSourceToContrastMask(const ImageGPU<3, byte> sourceImage, ImageGPU<1, byte> contrastMask, float minBrightness, float maxBrightness) {
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
        void KERNELContrastMaskToStrideMask(const ImageGPU<1, byte> contrastMask, ImageGPU<1, int16_t> strideMask, Direction direction) {
            Indices indices = contrastMask.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;

            bool vert = direction == Direction::vertical;

            bool isInStride{ false };
            int16_t strideLength{ 0 };
            Indices strideStartIndices;

            int count = vert ? contrastMask.getHeight() : contrastMask.getWidth();
            for (int i{ 0 }; i < count; ++i) {
                bool shouldThisPixelBeBlack{ true };
                bool passMask{ contrastMask.sample(indices)[0] != 0 };
                if (passMask) {
                    strideLength++;

                    // Start a stride
                    if (!isInStride) {
                        strideStartIndices = indices;
                        isInStride = true;
                        shouldThisPixelBeBlack = false;
                    }
                }
                else {
                    // Exit stride
                    if (isInStride) {
                        isInStride = false;
                        Pixel<1, int16_t> pixel;
                        pixel[0] = strideLength;
                        strideMask.setPixel(strideStartIndices, pixel);
                        strideLength = 0;
                    }
                }

                if (shouldThisPixelBeBlack) {
                    Pixel<1, int16_t> pixel;
                    pixel[0] = 0;
                    strideMask.setPixel(indices, pixel);
                }

                vert ? indices.second++ : indices.first++;
            }

            if (isInStride) {
                Pixel<1, int16_t> pixel;
                pixel[0] = strideLength;
                strideMask.setPixel(strideStartIndices, pixel);
                strideLength = 0;
            }
        }

        __global__
        void KERNELStrideMaskToNeatRGB8(const ImageGPU<1, int16_t> strideMask, ImageGPU<3, byte> displayImage, Direction direction) {
            Indices indices = strideMask.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;
            bool vert = direction == Direction::vertical;
            
            int16_t startingStrideLength{ 1 };
            int16_t strideLengthToGo{ 0 };
            
            int height = vert ? strideMask.getHeight() : strideMask.getWidth();
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
            
                vert ? (indices.second++) : (indices.first++);
            }
        }

        __global__
        void KERNELStrideMaskAndValuesToRGB8Sorted(const ImageGPU<1, int16_t> strideMask, const ImageGPU<1, __half> valueImage, ImageGPU<3, byte> sortedOutput) {
            Indices indices = strideMask.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;
            int16_t strideLength{ strideMask.sample(indices)[0] };
            if (strideLength == 0)
                return;
            
            extern __shared__ __half arrayToSort[];
        }
    }

    ImageCPU sortImage(const ImageCPU& sourceImage, Direction direction, float minContrast, float maxContrast, EffectType::Type type) {
        Timer<6> timer{{
            "Create values",
            "Create contrast",
            "Create stride",
            "Create smooth",
            "Move GPU image to CPU",
            "Move CPU image to GPU"
        }};
        constexpr int SQUARE_BLOCK_WIDTH{ 8 };
        constexpr int LINE_BLOCK_WIDTH{ 256 };

        dim3 squareThreadsPerBlock = dim3(SQUARE_BLOCK_WIDTH, SQUARE_BLOCK_WIDTH);
        dim3 squareBlockCount = dim3((sourceImage.getWidth() + SQUARE_BLOCK_WIDTH - 1) / SQUARE_BLOCK_WIDTH, (sourceImage.getWidth() + SQUARE_BLOCK_WIDTH - 1) / SQUARE_BLOCK_WIDTH);

        dim3 lineThreadsPerBlock;
        dim3 lineBlockCount;
        if (direction == Direction::vertical) {
            lineThreadsPerBlock = dim3(LINE_BLOCK_WIDTH, 1);
            lineBlockCount = dim3((sourceImage.getWidth() + LINE_BLOCK_WIDTH - 1) / LINE_BLOCK_WIDTH, 1);
        }
        else {
            lineThreadsPerBlock = dim3(1, LINE_BLOCK_WIDTH);
            lineBlockCount = dim3(1, (sourceImage.getHeight() + LINE_BLOCK_WIDTH - 1) / LINE_BLOCK_WIDTH);
        }

        timer.start(5);
        ImageGPU<3, byte> gpuSourceImage{ ImageGPU<3, byte>::copyFromCPU(sourceImage) };
        cudaDeviceSynchronize();
        timer.end(5);
        ImageGPU<1, __half> gpuValueImage{ ImageGPU<1, __half>::copyDimFromCPU(sourceImage) };
        ImageGPU<1, byte> gpuContrastMaskImage{ ImageGPU<1, byte>::copyDimFromCPU(sourceImage) };
        ImageGPU<1, int16_t> gpuStrideMaskImage{ ImageGPU<1, int16_t>::copyDimFromCPU(sourceImage) };
        ImageGPU<3, byte> outputImage{ ImageGPU<3, byte>::copyDimFromCPU(sourceImage) };

        timer.start(0);
        KERNELRGB8ToValue<<<squareBlockCount, squareThreadsPerBlock>>>(gpuSourceImage, gpuValueImage);
        cudaDeviceSynchronize();
        timer.end(0);

        timer.start(1);
        KERNELSourceToContrastMask<<<squareBlockCount, squareThreadsPerBlock>>>(gpuSourceImage, gpuContrastMaskImage, minContrast, maxContrast);
        cudaDeviceSynchronize();
        timer.end(1);

        if (type == EffectType::contrast) {
            KERNELOneValueToRGB8<<<squareBlockCount, squareThreadsPerBlock>>>(gpuContrastMaskImage, outputImage);
            return ImageCPU{ outputImage };
        }

        timer.start(2);
        KERNELContrastMaskToStrideMask<<<lineBlockCount, lineThreadsPerBlock>>>(gpuContrastMaskImage, gpuStrideMaskImage, direction);
        cudaDeviceSynchronize();
        timer.end(2);

        // KERNELStrideMaskAndValuesToRGB8Sorted<<<dim3(sourceImage.getWidth(), sourceImage.getHeight()), dim3(), >>>(gpuStrideMaskImage, gpuValueImage, gpuSortedImage);

        // Debugging
        timer.start(3);
        KERNELStrideMaskToNeatRGB8<<<lineBlockCount, lineThreadsPerBlock>>>(gpuStrideMaskImage, outputImage, direction);
        cudaDeviceSynchronize();
        timer.end(3);

        // KERNELOneValueToRGB8<<<squareBlockCount, squareThreadsPerBlock>>>(gpuStrideMaskImage, gpuSortedImage);

        timer.start(4);
        ImageCPU displayImage{ outputImage };
        cudaDeviceSynchronize();
        timer.end(4);
        timer.outputToFile("times.txt");
        return displayImage;
    }
}

#endif