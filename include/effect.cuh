#ifndef EFFECT_H
#define EFFECT_H

#include <string>
#include <string_view>
#include <array>
#include "pixel.cuh"
#include "imagecpu.h"
#include "effecttype.h"

namespace Effect {
    ImageCPU doEffect(EffectType::Type type, const ImageCPU& originalImage);
}

#endif