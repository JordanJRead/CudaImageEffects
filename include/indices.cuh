#ifndef INDICES_H
#define INDICES_H

class Indices {
public:
    Indices() = default;

    __host__ __device__
    Indices(size_t _x, size_t _y)
        : x{ _x }
        , y{ _y }
    {}

    __host__ __device__
    bool isValid() const {
        return x != (size_t)-1;
    }

    __host__ __device__
    Indices operator+(const Indices& other) const {
        return { x + other.x, y + other.y };
    }

    __host__ __device__
    Indices operator*(int s) const {
        return { x * s, y * s };
    }

    size_t x{};
    size_t y{};
};

#endif