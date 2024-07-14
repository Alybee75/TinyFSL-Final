# Changes from Teacher to Student Model Configurations

This document outlines the changes made between the teacher model and the student model in their respective configuration files. All teacher models can be seen inside the `fsl-tfl` folder.

## Training Configuration Changes
- `recognition_loss_weight`: Changed from `1.0` to `0.5`
- `translation_loss_weight`: Changed from `1.0` to `0.5`
- `epochs`: Changed from `5000000` to `100000`

## Model Configuration Changes
### Encoder
- `num_layers`: Changed from `3` to `2`
- `embeddings.embedding_dim`: Changed from `512` to `256`
- `hidden_size`: Changed from `512` to `256`
- `ff_size`: Changed from `2048` to `1024`

### Decoder
- `num_layers`: Changed from `3` to `2`
- `embeddings.embedding_dim`: Changed from `512` to `256`
- `hidden_size`: Changed from `512` to `246`
- `ff_size`: Changed from `2048` to `1024`
