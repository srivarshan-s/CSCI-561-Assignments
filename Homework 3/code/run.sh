#!/bin/sh

cp "data/train_data$1.csv" "train_data.csv"
cp "data/train_label$1.csv" "train_label.csv"
cp "data/test_data$1.csv" "test_data.csv"
cp "data/test_label$1.csv" "test_label.csv"

./venv/bin/python3 homework.py

rm "train_data.csv"
rm "train_label.csv"
rm "test_data.csv"
rm "test_label.csv"
