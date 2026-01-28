#ifndef EFFECT_H
#define EFFECT_H

#include <string>
#include <string_view>
#include <array>

namespace Effect {
    __device__
    Colour doInvert(const Colour& colour) {
        Colour invertedColour{ 1 - colour.getR(), 1 - colour.getG(), 1 - colour.getB() };
        return invertedColour;
    }

    __device__
    Colour doStipple(const Colour& colour, int pixelIndexX, int pixelIndexY) {
        float brightness = colour.getBrightness();

        int thresholdCount = 7;
        int threshold = (int)(brightness * thresholdCount);
        if (threshold >= thresholdCount)
            threshold = thresholdCount - 1;
        
        bool isWhite = false;
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
        return isWhite ? Colour{ 1.0f, 1.0f, 1.0f } : Colour{ 0.0f, 0.0f, 0.0f };
    }
}

#endif