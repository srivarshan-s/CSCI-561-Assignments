#!/bin/bash

# Compile the program
g++ -std=c++17 main.cpp -o main

# Run the compiled program
time ./main

# Remove the compiled program
rm main
