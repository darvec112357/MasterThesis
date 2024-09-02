#!/bin/bash

# Get the current date and time in the format YYYY-MM-DD_HH-MM-SS
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# Create a folder with the current date and time as its name
mkdir -p records/$current_time
cp preprocess.ipynb records/$current_time
cp train.py records/$current_time
# cp evaluate.py records/$current_time
python train.py gemma7 1 records/$current_time/gemma7
