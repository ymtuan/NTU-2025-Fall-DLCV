import torchvision.models as models
import torch.nn as nn
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dist_model(model_cfg):
    """
    Build a distance regressor model based on the provided configuration.
    
    Args:
        model_cfg (dict): Configuration dictionary containing model parameters.
    
    Returns:
        nn.Module: The constructed distance regressor model.
    """
    input_channels = model_cfg.get('input_channels', 5)
    backbone = model_cfg.get('backbone', 'resnet50')
    use_geometry = model_cfg.get('use_geometry', False)
    num_geo_features = model_cfg.get('num_geo_features', 14)

    # load model from path
    if 'model_path' in model_cfg:
        model_path = model_cfg['model_path']
        if not model_path.endswith('.pth'):
            raise ValueError("Model path must end with .pth")
        
        if use_geometry:
            model = GeometryFusedResNet(
                input_channels=input_channels, 
                num_geo_features=num_geo_features,
                backbone=backbone, 
                pretrained=False
            )
        else:
            model = ResNetDistanceRegressor(
                input_channels=input_channels, 
                backbone=backbone, 
                pretrained=False
            )
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
    
    return model


class GeometryFusedResNet(nn.Module):
    """
    Dual-Stream Architecture for Distance Regression:
    - Visual Stream: ResNet-50 backbone processing 6-channel input (RGB + Depth + 2 Masks)
    - Geometric Stream: MLP processing hand-crafted geometric features
    - Fusion Head: Combines both streams for final distance prediction
    """
    def __init__(self, input_channels=6, num_geo_features=14, backbone='resnet50', pretrained=True):
        super().__init__()
        
        # ===== Visual Stream (ResNet Backbone) =====
        self.resnet = getattr(models, backbone)(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify conv1 to accept 6 input channels (RGB + Depth + 2 Masks)
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize new conv1 weights
        if pretrained:
            with torch.no_grad():
                if input_channels >= 3:
                    # Copy RGB weights
                    self.resnet.conv1.weight[:, :3] = old_conv.weight
                if input_channels == 6:
                    # Initialize extra channels (Depth + 2 Masks)
                    # Use average of RGB weights for depth channel
                    self.resnet.conv1.weight[:, 3] = old_conv.weight.mean(dim=1)
                    # Initialize mask channels with small random values
                    nn.init.kaiming_normal_(self.resnet.conv1.weight[:, 4:], mode='fan_out', nonlinearity='relu')
        
        # Remove the original FC layer, keep only the feature extractor
        visual_feat_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Output: [B, 2048] for ResNet-50
        
        # ===== Geometric Stream (MLP) =====
        self.geo_mlp = nn.Sequential(
            nn.Linear(num_geo_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # ===== Fusion Head =====
        fusion_dim = visual_feat_dim + 128  # 2048 + 128 = 2176
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, geo_features):
        """
        Args:
            x: Visual input tensor of shape [B, 6, H, W]
            geo_features: Geometric features tensor of shape [B, num_geo_features]
        
        Returns:
            distance: Predicted distance of shape [B]
        """
        # Visual stream
        visual_feat = self.resnet(x)  # [B, 2048]
        
        # Geometric stream
        geo_feat = self.geo_mlp(geo_features)  # [B, 128]
        
        # Fusion
        fused = torch.cat([visual_feat, geo_feat], dim=1)  # [B, 2176]
        distance = self.fusion_head(fused)  # [B, 1]
        
        return distance.squeeze(1)  # [B]


class ResNetDistanceRegressor(nn.Module):
    """Baseline model without geometric features (backward compatible)"""
    def __init__(self, input_channels=5, backbone='resnet50', pretrained=False):
        super().__init__()
        self.resnet = getattr(models, backbone)(weights='IMAGENET1K_V1' if pretrained else None)

        # Replace first conv layer to accept custom number of input channels
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(input_channels, old_conv.out_channels,
                                      kernel_size=old_conv.kernel_size,
                                      stride=old_conv.stride,
                                      padding=old_conv.padding,
                                      bias=old_conv.bias is not None)

        # Copy weights if input_channels == 3
        if pretrained and input_channels == 6:
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3] = old_conv.weight
                self.resnet.conv1.weight[:, 3:] = old_conv.weight.clone()

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.resnet(x).squeeze(1)  # Output shape: (B,)
