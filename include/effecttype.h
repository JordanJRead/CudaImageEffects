#include <string>
#include <string_view>
#include <array>

namespace EffectType {
    enum Type {
        stipple,
        invert,
        max_effects
    };

    using namespace std::string_view_literals;
    constexpr std::array<std::string_view, max_effects> effectNames {
        "stipple"sv,
        "invert"sv
    };

    static_assert(effectNames.size() == max_effects, "You forgot to write all the effect names");

    Type getTypeFromString(std::string_view string) {
        for (size_t i{ 0 }; i < max_effects; ++i) {
            std::string_view effectName{ effectNames[i] };
            if (string == effectName)
                return (Type)i;
        }
        return max_effects;
    }

    std::string getPipedEffectNames() {
        std::string output{ "" };
        output += effectNames[0];
        for (size_t i{ 1 }; i < max_effects; ++i) {
            output += " | ";
            output += effectNames[i];
        }
        return output;
    }
}