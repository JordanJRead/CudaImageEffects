#include "../include/effect.cuh"
#include <string>
#include <string_view>
#include <array>
#include "../include/imagegpu.cuh"

namespace Effect {
    namespace {
        __global__
        void KERNALInvertImage(ImageGPU<3, byte> image) {
            Indices indices = image.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;

            Pixel<3, float> pixel{ image.sample(indices) };

            pixel[0] = 1 - pixel[0];
            pixel[1] = 1 - pixel[1];
            pixel[2] = 1 - pixel[2];

            image.setPixel(indices, Pixel<3, byte>{ pixel });
        }

                __global__
        void KERNALStippleImage(ImageGPU<3, byte> image) {
            Indices indices = image.getPixelIndices(blockDim, blockIdx, threadIdx);
            if (indices.first == (size_t)-1)
                return;

            Pixel<3, float> pixel{ image.sample(indices) };

            float brightness = pixel.getBrightness();
            int thresholdCount = 7;
            int threshold = (int)(brightness * thresholdCount);
            if (threshold >= thresholdCount)
                threshold = thresholdCount - 1;
            bool isWhite = false;

            size_t pixelIndexX{ indices.first };
            size_t pixelIndexY{ indices.second };
            
            switch(threshold) {
            case 6:
                isWhite = true;
                break;
            
            case 5:
                if (pixelIndexY % 3 == 0) {
                    isWhite = !(pixelIndexX % 4 == 0);
                }
                else if (pixelIndexY % 3 == 1) {
                    isWhite = !(pixelIndexX % 4 == 2);
                }
                else {
                    isWhite = true;
                }
                break;

            case 4:
                if (pixelIndexY % 2 == 0) {
                    isWhite = !(pixelIndexX % 4 == 0);
                }
                else {
                    isWhite = !(pixelIndexX % 4 == 2);
                }
                break;
            
            case 3:
                isWhite = !(pixelIndexX % 2 == pixelIndexY % 2);
                break;
            
            case 2:
                if (pixelIndexY % 2 == 0) {
                    isWhite = pixelIndexX % 4 == 0;
                }
                else {
                    isWhite = pixelIndexX % 4 == 2;
                }
                break;
            
            case 1:
                if (pixelIndexY % 3 == 0) {
                    isWhite = pixelIndexX % 4 == 0;
                }
                else if (pixelIndexY % 3 == 1) {
                    isWhite = pixelIndexX % 4 == 2;
                }
                else {
                    isWhite = false;
                }
                break;
            
            case 0:
                isWhite = false;
                break;
            }

            pixel[0] = isWhite;
            pixel[1] = isWhite;
            pixel[2] = isWhite;

            image.setPixel(indices, Pixel<3, byte>{ pixel });
        }
        
        ImageCPU doSimpleImageEffect(EffectType::Type type, const ImageCPU& originalImage) {
            ImageGPU<3, byte> gpuImage{ originalImage };

            constexpr int BLOCK_WIDTH{ 16 };
            dim3 blockDim = dim3(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 blockCount = dim3((gpuImage.getWidth() + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (gpuImage.getHeight() + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

            if (type == EffectType::invert)
                KERNALInvertImage<<<blockCount, blockDim>>>(gpuImage);
            if (type == EffectType::stipple)
                KERNALStippleImage<<<blockCount, blockDim>>>(gpuImage);
            ImageCPU alteredImage{ gpuImage };
            
            return alteredImage;
        }
    }

    ImageCPU doEffect(EffectType::Type type, const ImageCPU& originalImage) {
        return doSimpleImageEffect(type, originalImage);
    }
}