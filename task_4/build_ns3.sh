#!/bin/sh
CWD="$PWD"
echo "$CWD"
cd ~/Downloads/ns-allinone-3.30.1/ns-3.30.1/
cd ./scratch
#rm -rf ./*

cp "$CWD/$1.cc" ./
cd ..


./waf --run $1 $2 $3 $4 $5 $6
rm ./scratch/$1.cc
cd "$CWD"
