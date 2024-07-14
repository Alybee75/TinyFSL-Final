### Running Different CNN Models

To try different CNN models, you can easily switch between models by changing the checkpoint path listed under `load_model` in your YAML configuration file.

#### Example YAML Configuration (`config.yaml`):

```yaml
training:
    load_model: './ckpts/phoenix-train-models/{model}.ckpt'
```

- Replace `{model}` with the specific model name you want to use.
- Ensure that the checkpoint file (`{model}.ckpt`) exists in the specified directory (`./ckpts/phoenix-train-models/`). 
- Make sure to run `models.sh` in the root directory to download the models.

#### Steps:

1. **Navigate to Your YAML File**: Locate and open your YAML configuration file (`config.yaml`).

2. **Modify `load_model` Path**:
   - Update the `load_model` path to point to the desired checkpoint file for the CNN model you wish to use.
   - Example: To use a model named `vgg16`, modify the path to `./ckpts/phoenix-train-models/vgg16.ckpt`.

3. **Save Changes**: Save the YAML file after editing.

4. **Run Your Application**: After updating the YAML file with the new checkpoint path, proceed to run your application with the chosen CNN model.
