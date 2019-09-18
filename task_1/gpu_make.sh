#! /bin/bash

nvcc ./gpu/$1.cu -o ./bins/$1_g
