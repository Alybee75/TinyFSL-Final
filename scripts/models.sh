#!/bin/bash

# Create directory for checkpoints
mkdir -p ckpts
cd ckpts

# Create directory for phoenix-train and download models
mkdir -p phoenix-train-models
gdown --folder https://drive.google.com/drive/u/0/folders/1uNrXyTtEYhf658mVZ-mf7h773I5Myrf8 -O ./phoenix-train-models

# Create directory for transfer learning models and download
mkdir -p fsl-tfl-models
gdown --folder https://drive.google.com/drive/folders/1DHmVDcDqY8GajyCyeJxcoi5epJB6ctuw -O ./fsl-tfl-models

# Create directory for knowledge distillation models and download
mkdir -p fsl-kd-models
cd fsl-kd-models

# T = 1.5, a = 0.5
mkdir -p 1.5_0.5
gdown --folder https://drive.google.com/drive/folders/1j-EiE_6WGUIEvEYLtnucR0K5dtcriNQI -O ./1.5_0.5

# T = 3.0, a = 0.5
mkdir -p 3.0_0.5
gdown --folder https://drive.google.com/drive/folders/1AlMiOZwCOy1Pma5qI5FHmxlqVn0bmExU -O ./3.0_0.5
