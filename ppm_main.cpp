#include <stdio.h>

#include "io/ppm_handler.h"

#define X 100
#define Y 300

int main(){
    char img[X * Y];
    ppm_handler ppmmaker = ppm_handler(X, Y, 5);

    for(int i = 0; i < Y; i++){
        for(int j = 0; j < X; j++){
            img[i * X + j] = i % 10 * 25;
        }
    }

    ppmmaker.write_ppm("sample.ppm", img);

    return 0;
}