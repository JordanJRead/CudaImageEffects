#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <string_view>

typedef unsigned char byte;

class Image {
public:
    Image(std::string_view file);
    void writeToFile(std::string_view file) const;
    int getWidth() const { return mWidth; }
    int getHeight() const { return mHeight; }
    int getChannelCount() const { return mChannelCount; }
    int getByteCount() const { return mWidth * mHeight * mChannelCount; }
    bool getReadError() const { return mReadError; }
    bool getWriteError() const { return mWriteError; }
    const byte* getData() const { return mData; }
    ~Image();

private:
    bool mReadError{ false };
    bool mWriteError{ false };
    byte* mData;
    int mWidth;
    int mHeight;
    int mChannelCount;
};

#endif