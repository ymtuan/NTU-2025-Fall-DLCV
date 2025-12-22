import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import pycocotools.mask as mask_utils  # install pycocotools if needed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_inside_model(model_cfg):
    in_channels = model_cfg.get('input_channels', 5)
    use_geometry = model_cfg.get('use_geometry', False)
    num_geo_features = model_cfg.get('num_geo_features', 8)

    # load model from path
    if 'model_path' in model_cfg:
        model_path = model_cfg['model_path']
        if not model_path.endswith('.pth'):
            raise ValueError("Model path must end with .pth")
        
        if use_geometry:
            model = GeometryAwareInclusionModel(in_channels=in_channels, num_geo_features=num_geo_features)
        else:
            model = ResNet50Binary(in_channels=in_channels)
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
    
    return model

class ResNet50Binary(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        # Change input conv layer to accept in_channels
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Binary classification head
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
    def forward(self, x):
        x = self.resnet(x)
        return x.squeeze(1)  # [B] logits


class GeometryAwareInclusionModel(nn.Module):
    """
    Dual-Stream Architecture for Inclusion Classification:
    - Visual Stream: Modified ResNet50 extracting visual embeddings
    - Geometric Stream: MLP processing geometric features (IoU, depth diff, etc.)
    - Fusion Head: Combines both embeddings for final classification
    """
    def __init__(self, in_channels=5, num_geo_features=8, 
                 visual_embed_dim=128, geo_embed_dim=32):
        super().__init__()
        self.num_geo_features = num_geo_features
        self.visual_embed_dim = visual_embed_dim
        self.geo_embed_dim = geo_embed_dim
        
        # Visual Stream: Modified ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        # Change input conv layer to accept in_channels (RGB + Object Mask + Region Mask)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace final FC layer to output visual embeddings instead of logits
        resnet_fc_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(resnet_fc_in, visual_embed_dim)
        
        # Geometric Stream: MLP for geometric features
        self.geo_mlp = nn.Sequential(
            nn.Linear(num_geo_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, geo_embed_dim)
        )
        
        # 新增：幾何流自己的分類頭（輔助損失用）
        self.geo_aux_head = nn.Linear(geo_embed_dim, 1)
        
        # Fusion Head: Combines visual and geometric embeddings
        fusion_input_dim = visual_embed_dim + geo_embed_dim  # 128 + 32 = 160
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)
        )
        
    def forward(self, images, geo_features):
        """
        Args:
            images: [B, in_channels, H, W] - stacked RGB + masks
            geo_features: [B, num_geo_features] - geometric features (IoU, depth diff, etc.)
        
        Returns:
            main_logits: [B] - main classification logits from fusion head
            geo_logits: [B] - auxiliary logits from geometric stream (for auxiliary loss)
        """
        # Visual Stream
        visual_embed = self.resnet(images)  # [B, visual_embed_dim]
        
        # Geometric Stream
        geo_embed = self.geo_mlp(geo_features)  # [B, geo_embed_dim]
        
        # 輔助輸出：幾何流自己的分類頭
        geo_logits = self.geo_aux_head(geo_embed).squeeze(1)  # [B]
        
        # Fusion
        combined = torch.cat([visual_embed, geo_embed], dim=1)  # [B, 160]
        main_logits = self.fusion_head(combined).squeeze(1)  # [B]
        
        return main_logits, geo_logits