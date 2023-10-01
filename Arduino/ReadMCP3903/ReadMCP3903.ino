#include "MCP3903.hpp"

"""
IMPORTANT:
When the board disconnects for any reason, it has to be reset, otherwise the registers will not be set correctly and therefore reading will be impossible.
"""

#define CS 53
#define RESET 9

#define DATAREADY_A 2
#define DATAREADY_B 6
#define DATAREADY_C 7

#define CHANNELS 12

MCP3903 adc(CS,RESET,DATAREADY_A,DATAREADY_B,DATAREADY_C);

#include <SPI.h>

const unsigned long READ_NUM = 20;
const unsigned long READ_DELAY = 100000/READ_NUM;

unsigned long last_ready = 0;
int count = 0;
bool ready = false;

void setup() {
  Serial.begin(2000000);
  adc.begin();
  attachInterrupt(digitalPinToInterrupt(DATAREADY_A),interrupt,FALLING);
  last_ready = micros();
}

void loop() {
  if(Serial.available()){
    String Serial_in = Serial.readString();
    if(Serial_in[0]=='p'){
      dataForProcessing();
    }
  }
}

void interrupt(){
  ready = true;
}

void dataForProcessing(){
  boolean reset = false;
  unsigned long start_start_time = millis();
  unsigned long start_time = micros();
  unsigned long i = 0;
  while(true){
    if(checkForReset()){
      break;
    }
    //+1000 to make sure that 1ms passes after last datapoint, before first datapoint respectively
    if(micros()-start_time>=i*READ_DELAY+READ_DELAY){
      if(ready){
        ready = false;

        int num_bits = 4;
        char outputArray[num_bits*CHANNELS];

        uint32_t dataArray[CHANNELS];
        adc.getData(dataArray,CHANNELS);
        for(int l=0;l<CHANNELS;l++){
          // uint32_t number = 16777216*((millis()-start_start_time+l*500)/5000.);
          uint32_t number = dataArray[l];
          byte bitMask = 0b00111111;
          // if(l==1)
          //   Serial.println(number);
          outputArray[l*num_bits + 0] = ((number >> 18) & bitMask) + 64; // Most significant byte (MSB)
          outputArray[l*num_bits + 1] = ((number >> 12) & bitMask) + 64;  // Middle byte
          outputArray[l*num_bits + 2] = ((number >> 6) & bitMask) + 64;  // Middle byte
          outputArray[l*num_bits + 3] = (number & bitMask) + 64;         // Least significant byte (LSB)

          // also invert 24th bit, to have negative in range 0-2^23 and positive in range 2^23-2^24
          int bitToInvert = 5;

          // Invert the selected bit using XOR (^) operation with a bitmask
          outputArray[l*num_bits + 0] ^= (1 << bitToInvert);
        }

        Serial.write(outputArray, num_bits*CHANNELS);
        // Serial.print(outputArray[4],BIN); Serial.print(" "); Serial.print(outputArray[5],BIN); Serial.print(" "); Serial.print(outputArray[6],BIN); Serial.print(" "); Serial.print(outputArray[7],BIN); Serial.print(" "); Serial.println(dataArray[1]);

        i++;
      }
      //print line after READ_NUM datapoints
      if(i%READ_NUM==0){
        Serial.println();
      }
      //reset i to avoid overflow issues
      if(i%(READ_NUM*10)==0){
        i = 0;
        start_time = micros();
      }
    }

    if((millis()-start_start_time)>=5000)
      start_start_time = millis();
  }
}

boolean checkForReset(){
  if(Serial.available()){
    if(Serial.readString()[0]=='r'){
      return true;
    }
  }
  return false;
}