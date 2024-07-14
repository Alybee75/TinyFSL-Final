# Freeze all parameters first

checkpoint = torch.load('path_to_your_ckpt_file.ckpt', map_location='cpu')
model_state = checkpoint['model_state']

for param in fsl_model.parameters():
    param.requires_grad = False

# Unfreeze the top layers (gloss output layer and decoder output layer)
if hasattr(fsl_model, 'gloss_output_layer'):
    for param in fsl_model.gloss_output_layer.parameters():
        param.requires_grad = True

if hasattr(fsl_model.decoder, 'output_layer'):
    for param in fsl_model.decoder.output_layer.parameters():
        param.requires_grad = True
