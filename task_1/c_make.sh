#! /bin/bash
if [[ $2 == "NO" ]]
then
    gcc ./c_files/$1.c -o ./bins/$1
else
    gcc -O3 -mavx ./c_files/$1.c -o ./bins/$1
fi
