#! /bin/bash

nvcc -O3 ./gpu/$1.cu -o ./bins/$1_g
