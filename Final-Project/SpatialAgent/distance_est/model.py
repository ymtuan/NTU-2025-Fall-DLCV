import torchvision.models as models
import torch.nn as nn
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dist_model(model_cfg):
    """
    Build a distance regressor model based on the provided configuration.
    
    Args:
        model_cfg (dict): Configuration dictionary containing model parameters.
            - model_path: Path to checkpoint file
            - use_geometry: Whether to use geometric features (for GeometryFusedResNet)
            - use_shortcut: Whether to use ResNetWithShortcut (simple shortcut architecture)
            - input_channels: Number of input channels (default: 5 or 6)
            - num_geo_features: Number of geometric features (default: 14 for GeometryFusedResNet, 3 for ResNetWithShortcut)
            - backbone: Backbone architecture (default: 'resnet50')
    
    Returns:
        nn.Module: The constructed distance regressor model.
    """
    input_channels = model_cfg.get('input_channels', 5)
    backbone = model_cfg.get('backbone', 'resnet50')
    use_geometry = model_cfg.get('use_geometry', False)
    use_shortcut = model_cfg.get('use_shortcut', False)
    num_geo_features = model_cfg.get('num_geo_features', 14 if not use_shortcut else 3)

    # load model from path
    if 'model_path' in model_cfg:
        model_path = model_cfg['model_path']
        if not model_path.endswith('.pth'):
            raise ValueError("Model path must end with .pth")
        
        if use_shortcut:
            # ResNetWithShortcut: Simple shortcut architecture with 3 geometric features
            model = ResNetWithShortcut(
                in_channels=input_channels,
                num_geo_features=num_geo_features,
                backbone=backbone,
                pretrained=False
            )
        elif use_geometry:
            # GeometryFusedResNet: Dual-stream architecture with 14 geometric features
            model = GeometryFusedResNet(
                input_channels=input_channels, 
                num_geo_features=num_geo_features,
                backbone=backbone, 
                pretrained=False
            )
        else:
            # ResNetDistanceRegressor: Baseline model without geometric features
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


class ResNetWithShortcut(nn.Module):
    """
    ResNet model with geometric features shortcut connection.
    Directly concatenates geometric features (mean_depth_1, mean_depth_2, centroid_dist_2d)
    to the visual features before the final FC layer.
    """
    def __init__(self, in_channels=6, num_geo_features=3, backbone='resnet50', pretrained=True):
        super().__init__()
        # Backbone
        self.resnet = getattr(models, backbone)(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify first layer conv1 to accept custom input channels
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize new conv1 weights
        if pretrained:
            with torch.no_grad():
                if in_channels >= 3:
                    # Copy RGB weights
                    self.resnet.conv1.weight[:, :3] = old_conv.weight
                if in_channels == 6:
                    # Initialize extra channels (Depth + 2 Masks)
                    # Use average of RGB weights for depth channel
                    self.resnet.conv1.weight[:, 3] = old_conv.weight.mean(dim=1)
                    # Initialize mask channels with small random values
                    nn.init.kaiming_normal_(self.resnet.conv1.weight[:, 4:], mode='fan_out', nonlinearity='relu')
        
        # Modify last layer
        num_visual_feats = self.resnet.fc.in_features  # 2048
        num_geo_feats = num_geo_features  # 3: [depth_a, depth_b, 2d_dist]
        
        # Remove original fc
        self.resnet.fc = nn.Identity()
        
        # New Head: combine visual + geometric features
        self.fusion_head = nn.Sequential(
            nn.Linear(num_visual_feats + num_geo_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, img, geo_feats):
        """
        Args:
            img: Visual input tensor of shape [B, 6, H, W]
            geo_feats: Geometric features tensor of shape [B, 3] containing
                      [mean_depth_1, mean_depth_2, centroid_dist_2d]
        
        Returns:
            distance: Predicted distance of shape [B]
        """
        # 1. Visual features
        visual_emb = self.resnet(img)  # [B, 2048]
        
        # 2. Geometric features (should be normalized to [0, 1] in Dataset)
        # geo_feats: [B, 3]
        
        # 3. Concatenate
        combined = torch.cat([visual_emb, geo_feats], dim=1)  # [B, 2051]
        return self.fusion_head(combined).squeeze(1)  # [B]


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
