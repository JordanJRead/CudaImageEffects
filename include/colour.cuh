#ifndef COLOUR_H
#define COLOUR_H

typedef unsigned char byte;

class Colour {
public:
    __host__ __device__
    Colour();

    __host__ __device__
    Colour(byte r, byte g, byte b);
    
    __host__ __device__
    Colour(float r, float g, float b);
    
    __host__ __device__
    float getBrightness() const;

    __host__ __device__
    float getR() const { return mR; }
    __host__ __device__
    float getG() const { return mG; }
    __host__ __device__
    float getB() const { return mB; }
    
    __host__ __device__
    byte getByteR() const { return mR * 255; }
    __host__ __device__
    byte getByteG() const { return mG * 255; }
    __host__ __device__
    byte getByteB() const { return mB * 255; }
    
    __host__ __device__
    float operator*(const Colour& other) const;

private:
    float mR;
    float mG;
    float mB;
};

#endif