import torch
import torchvision.transforms as T

# Try timm first, fall back to clip
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

try:
    import clip
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

from PIL import Image

class ViTEncoder:
    """
    Wrap a pretrained ViT that returns visual token embeddings of shape (B, v_len, vision_dim).
    By default returns patch tokens (without cls) to keep a few visual tokens.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.vision_dim = None
        if _HAS_TIMM:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="")
            self.model.eval().to(device)
            # timm ViT exposes forward_features -> (B, num_patches+1, embed_dim)
            self.vision_dim = self.model.embed_dim
            self.preprocess = T.Compose([
                T.Resize(224, interpolation=Image.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            ])
        elif _HAS_CLIP:
            self.model, clip_pre = clip.load("ViT-B/16", device=device, jit=False)
            self.model.eval().to(device)
            # CLIP returns pooled features but we can use visual.transformer to get patch tokens if available
            self.vision_dim = self.model.visual.output_dim
            self.preprocess = clip_pre
        else:
            raise RuntimeError("Neither timm nor clip available. Install one to extract visual features.")

    @torch.no_grad()
    def extract(self, pil_images, return_cls=False):
        """
        Args:
          pil_images: list of PIL.Image
        Returns:
          vision_tokens: (B, v_len, vision_dim)
        """
        images = [self.preprocess(im).unsqueeze(0) if isinstance(self.preprocess, T.Compose) else self.preprocess(im).unsqueeze(0) for im in pil_images]
        x = torch.cat(images, dim=0).to(self.device)
        if _HAS_TIMM:
            # timm ViT: forward_features returns (B, num_patches+1, dim)
            feats = self.model.forward_features(x)  # (B, N+1, D)
            # drop cls token (index 0) to get patch tokens
            if feats.shape[1] > 1 and not return_cls:
                return feats[:, 1:, :].contiguous()
            return feats.contiguous()
        else:
            # CLIP: try to access transformer to get patch tokens
            if hasattr(self.model.visual, "transformer"):
                x = self.model.visual.conv1(x.type(self.model.visual.conv1.weight.dtype))
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1)
                x = torch.cat([self.model.visual.cls_token.repeat(x.shape[0],1,1), x], dim=1)
                x = x + self.model.visual.positional_embedding
                x = self.model.visual.transformer(x)
                if not return_cls:
                    return x[:, 1:, :].contiguous()
                return x.contiguous()
            else:
                # fallback: use pooled features expanded to single token
                pooled = self.model.encode_image(x)
                return pooled.unsqueeze(1).contiguous()
