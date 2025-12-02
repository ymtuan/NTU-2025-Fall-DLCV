import torch
import segmentation_models_pytorch as smp

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=7)
x = torch.randn(1, 3, 256, 256)
features = model.encoder(x)
for i, f in enumerate(features):
    print(i, f.shape)
