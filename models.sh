#!/bin/bash

# Create directory for checkpoints
mkdir -p ckpts
cd ckpts

# Create directory for phoenix-train and download models
mkdir -p phoenix-train-models
wget --folder https://drive.google.com/drive/u/0/folders/1uNrXyTtEYhf658mVZ-mf7h773I5Myrf8 -O ./phoenix-train-models

