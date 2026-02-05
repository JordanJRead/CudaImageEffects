#include "stdio.h"
#include <string>
#include <string_view>
#include <array>
#include <filesystem>
#include "include/effect.cuh"
#include "include/imagecpu.h"
#include "include/effecttype.h"
#include "include/timer.h"

typedef unsigned char byte;

/*
ARGS:
0: Program name
1: Input file
2: Effect name
3: [Contrast min]
4: [Contrast max]
*/
int main(int argc, char* argv[]) {
    // Input validation
    if (argc != 3 && argc != 5) {
        printf("Usage: ./main (input_image_file) (%s) [contrast_min] [contrast_max]\ncontrast_min/max are used for sort_vert, sort_hor, and contrast.", EffectType::getPipedEffectNames().c_str());
        return 1;
    }

    std::filesystem::path filePath{ argv[1] };
    std::filesystem::path outputFilePath;
    std::string effectTypeInput{ argv[2] };
    
    float minContrast = 0.3;
    float maxContrast = 0.7;
    if (argc == 5) {
        try {
            minContrast = std::stof(std::string{ argv[3] });
            maxContrast = std::stof(std::string{ argv[4] });
            if (minContrast >= maxContrast) {
                printf("Min contrast must be less than max contrast\n");
                return 1;
            }
        }
        catch (...) {
            printf("Could not parse contrast values\n");
            return 1;
        }
    }

    EffectType::Type effectType{ EffectType::getTypeFromString(effectTypeInput) };
    if (effectType == EffectType::max_effects) {
        printf("Invalid effect name used, must be one of: %s\n", EffectType::getPipedEffectNames().c_str());
        return 1;
    }
    std::string outputFilePathName{ filePath.parent_path().string().c_str() };
    outputFilePathName += "/";
    outputFilePathName += filePath.stem().string().c_str();
    outputFilePathName += "_";
    outputFilePathName += EffectType::effectNames[effectType];
    outputFilePathName += ".png";
    outputFilePath = outputFilePathName;

    // Load image data
    ImageCPU originalImage{ filePath.string().c_str() };
    if (originalImage.getReadError()) {
        return 1;
    }

    ImageCPU alteredImage{ Effect::doEffect(effectType, originalImage, minContrast, maxContrast) };
    alteredImage.writeToFile(outputFilePath.string().c_str());

    if (alteredImage.getWriteError()) {
        printf("Error writting to output file\n");
        return 1;
    }
    return 0;
}