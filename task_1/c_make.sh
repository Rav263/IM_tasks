#! /bin/bash

gcc -O3 -mavx ./c_files/$1.c -o ./bins/$1
