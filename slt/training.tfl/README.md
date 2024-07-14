# Model Training for Phoenix14T Dataset and Transfer Learning for FSL Dataset

This folder is dedicated to running model training for the Phoenix14T dataset and transfer learning tasks. The configurations for these tasks can be found in the following directories:

- `configs/phoenix`: Configuration files for training models on the Phoenix14T dataset.
- `configs/fsl-tfl`: Configuration files for transfer learning tasks.

## Model Training

To run the model training tasks, please follow these steps:

1. **Prepare the Environment**: Ensure you have set up your environment as described in the main setup instructions.

2. **Run the Training Script**: Use the appropriate training script and configuration file to start the training process. For example, you can use the following command to train a model on the Phoenix14T dataset:

    ```bash
    python -m slt.training.tfl train configs/phoenix/squeezenet.yaml
    ```

    This sample command can be found in the `scripts/train_model.sh` file for reference.

## Transfer Learning

To run the transfer learning tasks, please follow these steps:

1. **Prepare the Environment**: Ensure you have set up your environment as described in the main setup instructions.

2. **Update the YAML Configuration File**: Before running the transfer learning scripts, make sure to update the `load_model` parameter in your YAML configuration file. This parameter should point to the checkpoint (`ckpt`) of the model you want to learn from. 

    Example:
    ```yaml
    training:
        load_model: path/to/your/model_checkpoint.ckpt
    ```

3. **Run the Training Script**: Use the appropriate training script and configuration file to start the training process. For example, you can use the following command to run a transfer learning task:

    ```bash
    python -m slt.training.tfl train configs/fsl-tfl/combined.yaml
    ```

    This sample command can be found in the `scripts/train_model.sh` file for reference.
