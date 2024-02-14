#!/bin/bash

# Compile the homework.cpp file
g++ -std=c++17 homework.cpp -o main

# Run the compiled program
time ./main

# Remove the compiled file
rm main
