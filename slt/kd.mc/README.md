# Knowledge Distillation and Model Compression

This folder is dedicated to running knowledge distillation and model compression tasks. The configurations for these tasks can be found in the `configs/fsl-kd` directory.

## Knowledge Distillation

To perform knowledge distillation, please follow these steps:

1. **Edit the `__main__.py` File**: Before running the distillation process, edit the `__main__.py` file under the `kd.mc` folder. Ensure you place the config file of the teacher model in `cfg_file` at line 43.

2. **Adjust Temperature and Alpha**: Modify the `Temperature` and `Alpha` values under the `_train_batch` function in `training.py` and under the `validate_on_data` function in `prediction.py` to your desired settings.

3. **Run the Distillation Script**: The config file should be for the student model, and the `ckpt` model should be the TFL model. You can download the TFL model checkpoint by running the `scripts/models.sh` script or by executing the transfer learning commands and obtaining the best checkpoint (`best.ckpt`). Use the following command to run the knowledge distillation:

    ```bash
    python -m slt.kd.mc train configs/fsl-kd/combined.yaml --ckpt fsl-tfl-models/combined_best.ckpt
    ```

    This sample command can be found in the `scripts/train_model.sh` file for reference.

## Model Conversion to TFlite

After completing the knowledge distillation, follow these steps to perform model conversion:

1. **Identify the Best Checkpoint**: The checkpoint (`ckpt`) should be the best model obtained after training with knowledge distillation. You can download it by running the `scripts/models.sh` script or by executing the knowledge distillation command above and obtaining the best checkpoint (`best.ckpt`).

2. **Run the Conversion Script**: Use the following command to convert the model:

    ```bash
    python -m slt.kd.mc convert configs/fsl-kd/combined.yaml --ckpt "fsl-kd-models/3.0_0.5/combined.ckpt"
    ```

    This sample command can be found in the `scripts/train_model.sh` file for reference.
