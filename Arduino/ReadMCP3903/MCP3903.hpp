#ifndef MCP3903_HPP
#define MCP3903_HPP

#define ADC_AIN9 B1110
#define ADC_AIN10 B1111

#include "Arduino.h"
#include <SPI.h>


class MCP3903{
    public:
        MCP3903(int csPin, int resetPin, int readyPinA, int readyPinB,int readyPinC);
        void begin();
        void getData(uint32_t *data, int num_channels);
    private:
        int _csPin;
        int _resetPin;
        int _readyPinA;
        int _readyPinB;
        int _readyPinC;
        SPISettings _spiSettings;
};

#endif