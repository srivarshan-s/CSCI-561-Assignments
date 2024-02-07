#!/bin/bash

# Compile the source code
clang++ homework.cpp -o main
#
# Loop through files from input1.txt to input50.txt
for ((i = 1; i <= 50; i++)); do
    # Append the contents of the file to input.txt
    cat "../grading-testcases/input${i}.txt" > input.txt

    # Run the binary
    ./main > /dev/null 2>&1

    # Print the actual pathlen
    echo "Testcase ${i}"
    cat "pathlen.txt" 
    cat "../grading-testcases/pathlen${i}.txt" 
    echo " "
done
