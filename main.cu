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

int main(int argc, char* argv[]) {
    Timer<1> timer;
    timer.start(0);
    // Input validation
    if (argc != 3 && argc != 4) {
        printf("Usage: ./main (%s) (input_image_file) [output_image_file]", EffectType::getPipedEffectNames().c_str());
        return 1;
    }

    std::string effectTypeInput{ argv[1] };
    std::filesystem::path filePath{ argv[2] };
    std::filesystem::path outputFilePath;

    EffectType::Type effectType{ EffectType::getTypeFromString(effectTypeInput) };
    if (effectType == EffectType::max_effects) {
        printf("Usage: ./main (%s) (input_image_file) [output_image_file]", EffectType::getPipedEffectNames().c_str());
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
    ImageCPU originalImage{ filePath.string().c_str() };
    if (originalImage.getReadError()) {
        return 1;
    }

    ImageCPU alteredImage{ Effect::doEffect(effectType, originalImage) };
    alteredImage.writeToFile(outputFilePath.string().c_str());

    if (alteredImage.getWriteError()) {
        return 1;
    }
    timer.end(0);
    timer.outputToFile("times.txt");
    return 0;
}