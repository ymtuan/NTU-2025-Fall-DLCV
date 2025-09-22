import segmentation_models_pytorch as smp
import torch

class UnetSkip(smp.Unet):
    def __init__(self, *args, drop_idx=None, **kwargs):
        """
        drop_idx: which encoder feature to drop (int or list of int)
                  e.g. 1 = first skip, 2 = second skip, etc.
        """

        super().__init__(*args, **kwargs)
        if drop_idx is None:
            drop_idx = []
        elif isinstance(drop_idx, int):
            drop_idx = [drop_idx]
        self.drop_idx = drop_idx
    
    def forward(self, x):
        features = self.encoder(x)
        # Drop delected skip ocnnections
        for idx in self.drop_idx:
            if 0 <= idx < len(features):
                features[idx] = torch.zeros_like(features[idx])
            
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks
