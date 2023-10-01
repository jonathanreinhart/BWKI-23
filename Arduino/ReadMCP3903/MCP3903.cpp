#include "MCP3903.hpp"

MCP3903::MCP3903(int csPin, int resetPin, int readyPinA, int readyPinB,int readyPinC){
    _csPin = csPin;
    _resetPin = resetPin;
    _readyPinA = readyPinA;
    _readyPinB = readyPinB;
    _readyPinC = readyPinC;
    _spiSettings = SPISettings(1000000, MSBFIRST, SPI_MODE0);
}

void MCP3903::begin(){
    pinMode(_csPin,OUTPUT);
    pinMode(_resetPin,OUTPUT);
    pinMode(_readyPinA,INPUT_PULLUP);
    pinMode(_readyPinB,INPUT_PULLUP);
    pinMode(_readyPinC,INPUT_PULLUP);

    digitalWrite(_csPin,HIGH);

    //reset device
    digitalWrite(_resetPin,LOW);
    delayMicroseconds(20);
    digitalWrite(_resetPin,HIGH);
    delayMicroseconds(20);

    SPI.begin();
    SPI.beginTransaction(_spiSettings);
    //setup device
    //first byte is always the Communications Register
    //with last four bits we specify the register we want to write to
    //the second bit is the read/write bit
    
    // SPI.transfer(0b01010011);
    // uint32_t output1 = SPI.transfer16(0xffff);
    // byte output2 = SPI.transfer(0xff);
    // uint32_t output = 0;
    // output = output1<<8|output2;
    //          100000000100000000111111
    // default: 100000000100000000000000

    //          000000000000111111010000
    // default: 000000000000111111010000
    // Serial.print("output1: "); Serial.println(output1,BIN);
    // Serial.print("output2: "); Serial.println(output2,BIN);
    // Serial.print("Status register: "); Serial.println(output,BIN);

    // configure the device
    // the default adress looping is set to type so we 
    // can write to all registers in one go

    digitalWrite(_csPin,LOW);
    delayMicroseconds(10);

    // go to gain register
    SPI.transfer(0b01010000);

    // set gain to 1 an current to x1 (set this to x2 for high speed mode)
    SPI.transfer16(0);
    SPI.transfer(0);

    // configure the communication register
    SPI.transfer16(0b1001111111110000);
    SPI.transfer(0b00111111);

    // configure the config register
    SPI.transfer16(0b0000000000001111);
    SPI.transfer(0b11110000);
    digitalWrite(_csPin,HIGH);

    delayMicroseconds(20);

    digitalWrite(_csPin,LOW);
    delayMicroseconds(10);

    // // read all control registers
    // SPI.transfer(0b01001101);
    // // 100111111111000000111111

    // for(int i=0;i<5;i++){
    //     uint32_t output1 = SPI.transfer16(0xffff);
    //     byte output2 = SPI.transfer(0xff);
    //     uint32_t output = output1<<8|output2;
    //     Serial.print("register "); Serial.print(i); Serial.print(": "); Serial.println(output,BIN);
    // }

    // digitalWrite(_csPin,HIGH);
    // delayMicroseconds(20);
}

// info: don't use chip select because to slow
void MCP3903::getData(uint32_t *data, int num_channels){
    digitalWrite(_csPin,LOW);
    delayMicroseconds(10);

    // go to ch0 register in read mode
    SPI.transfer(0b01000001);

    // read data for each channel
    for(int i=0;i<num_channels;i++){
        // uint32_t output1 = 0;
        uint32_t output1 = SPI.transfer16(0xffff);
        // byte output2 = 0;
        byte output2 = SPI.transfer(0xff);
        // uint32_t output = 0;
        // Serial.println("-----------------");
        // Serial.println(output1,BIN);
        // Serial.println(output2,BIN);
        uint32_t output = output1<<8|output2;
        // Serial.println(output,BIN);
        // Serial.println("-----------------");
        data[i] = output;
        // if(i==1)
        //   Serial.println(data[i]);
        //        1110101001011000101100
        // 110101001101010011010101101001
    }
    digitalWrite(_csPin,HIGH);
}