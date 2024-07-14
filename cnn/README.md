# Data Preparation

## Pretrained Models

The datasets were generated using pretrained models of MobileNet, InceptionV4, SqueezeNet, and AlexNet. The feature extraction and processing were performed using these models to ensure high-quality and robust feature representations.

## Dataset Format

The dataset format used in this project is as follows:

- **Format**: The datasets are stored in gzipped pickle files.
- **Structure**: Each gzipped pickle file contains a list of dictionaries. Each dictionary represents a single data sample with the following keys:
  - `sign`: The extracted features from the video, stored as a tensor.
  - `gloss`: The gloss of the sign.
  - `text`: The text translation of the sign.
  - `name`: The name or path of the video sequence.
  - `signer`: The signer information.

## Example Output

An example output of this dataset format, when extracted into a text file, can be seen in `sample_output.txt`.