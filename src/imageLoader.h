typedef unsigned char byte; // most useful typedef ever

// see docs in imageLoader.cpp
struct imgData {
    imgData(byte* pix = nullptr, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {
    };
    byte* pixels;
    unsigned int width;
    unsigned int height;
};

struct imgDataF {
    imgDataF(float* pix = nullptr, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {
    };
    float* pixels;
    unsigned int width;
    unsigned int height;
};

imgData loadImage(char* filename);
void writeImage(char* filename, std::string appendTxt, imgData img);

void byteToFloat(imgData img_byte, imgDataF img_float, int);
void floatToByte(imgDataF img_float, imgData img_byte, int);
