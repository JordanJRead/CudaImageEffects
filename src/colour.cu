#include "../include/colour.cuh"

__host__ __device__
Colour::Colour()
    : mR{ 0 }
    , mG{ 0 }
    , mB{ 0 }
{}

__host__ __device__
Colour::Colour(byte r, byte g, byte b)
    : mR{ r / 255.0f }
    , mG{ g / 255.0f }
    , mB{ b/ 255.0f }
{}

__host__ __device__
Colour::Colour(float r, float g, float b)
    : mR{ r }
    , mG{ g }
    , mB{ b }
{}

__host__ __device__
float Colour::getBrightness() const {
    return *this * Colour{0.299f, 0.587f, 0.114f};
}

__host__ __device__
float Colour::operator*(const Colour& other) const {
    return mR * other.mR + mG * other.mG + mB * other.mB;
}