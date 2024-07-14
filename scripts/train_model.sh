#!/bin/bash

# Use the absolute path to your project directory
PROJECT_PATH="$(pwd)"

# Set PYTHONPATH to include the "slt" directory within your project
export PYTHONPATH="${PROJECT_PATH}/slt:${PYTHONPATH}"

# Run your Python module command to train using a specific YAML configuration file

# Model Training
python -m slt.training.tfl train configs/phoenix/squeezenet.yaml

# Transfer Learning
# Make sure to change the `load_model` in your YAML configuration file and load the ckpt of the
# model you want to learn from.
python -m slt.training.tfl train configs/fsl-tfl/combined.yaml

# Knowledge Distillation
# Before running, edit the `__main__.py file under the kd.mc folder. Make sure to place the config file
# of the teacher model in cfg_file at line 43.
# Also, make sure to change the Temperature and the Alpha under the _train_batch function in `training.py`;
# and also under the validate_on_data function in `prediction.py`.

# The config file here should be the student model, and the ckpt model here should be the TFL model.
# You can download it by running scripts/models.sh or by running the code above under Transfer
# Learning and getting the best.ckpt
python -m slt.kd.mc train configs/fsl-kd/combined.yaml --ckpt fsl-tfl-models/combined_best.ckpt

# Model Conversion
# The ckpt here should be the best model after training with knowledge distillation. You can download 
# it by running scripts/models.sh or by running the code above under Knowledge Distillation and 
# getting the best.ckpt
python -m slt.kd.mc convert configs/fsl-kd/combined.yaml --ckpt "fsl-kd-models/3.0_0.5/combined.ckpt"
