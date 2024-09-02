#!/bin/bash

for strategy in "cot" "pot" "sub"; do
    python infer.py gemma7 results/gemma7 $strategy 1
done
    
