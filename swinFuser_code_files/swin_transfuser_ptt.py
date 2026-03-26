import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class SwinPTTBackbone(nn.Module):
    """
    Fixed Multi-scale Fusion using Swin Transformer for PTT-processed LiDAR + CNN-processed images
    
    Key fixes:
    - Spatial dimension alignment before fusion
    - Proper feature dimension handling
    - Compatible with existing TransFuser training pipeline
    - FIXED: Attention head dimensions to match channel dimensions
    """

    def __init__(self, config, image_architecture='resnet34', lidar_architecture='resnet18', use_velocity=True):
        super().__init__()
        self.config = config
        self.use_velocity = use_velocity

        # Use unified spatial dimensions for fusion
        self.fusion_size = (8, 8)  # Common size for both modalities
        self.unified_pool = nn.AdaptiveAvgPool2d(self.fusion_size)
        
        # CNN-based image encoder
        self.image_encoder = ImageCNN(
            architecture=image_architecture, 
            normalize=True
        )

        # Determine LiDAR input channels
        if config.use_point_pillars:
            in_channels = config.num_features[-1]
        else:
            in_channels = 2 * config.lidar_seq_len
            
        if getattr(config, 'use_target_point_image', False):
            in_channels += 1

        # LiDAR encoder (standard CNN for compatibility)
        self.lidar_encoder = LidarEncoder(
            architecture=lidar_architecture, 
            in_channels=in_channels
        )

        # Get feature dimensions from encoders
        try:
            img_feature_dims = [
                self.image_encoder.features.feature_info[i]['num_chs'] 
                for i in range(1, 5)
            ]
        except:
            # Fallback for architectures without feature_info
            img_feature_dims = [64, 128, 256, 512]
        
        # FIXED: Ensure all num_heads values evenly divide the channel dimensions
        # For typical ResNet: [64, 128, 256, 512]
        # Good head counts: 64->4, 128->8, 256->8, 512->8
        
        self.swin_fusion1 = SwinFusionModule(
            img_dim=img_feature_dims[0], 
            lidar_dim=img_feature_dims[0],  # Assume same for simplicity
            fusion_size=self.fusion_size,
            window_size=4,  # Smaller window for small features
            num_heads=4,  # 64/4=16 ✓
            use_velocity=use_velocity
        )
        
        self.swin_fusion2 = SwinFusionModule(
            img_dim=img_feature_dims[1], 
            lidar_dim=img_feature_dims[1],
            fusion_size=self.fusion_size,
            window_size=4,
            num_heads=8,  # FIXED: 128/8=16 ✓ (was 6, which gave 128/6=21.33)
            use_velocity=use_velocity
        )
        
        self.swin_fusion3 = SwinFusionModule(
            img_dim=img_feature_dims[2], 
            lidar_dim=img_feature_dims[2],
            fusion_size=self.fusion_size,
            window_size=4,
            num_heads=8,  # 256/8=32 ✓
            use_velocity=use_velocity
        )
        
        self.swin_fusion4 = SwinFusionModule(
            img_dim=img_feature_dims[3], 
            lidar_dim=img_feature_dims[3],
            fusion_size=self.fusion_size,
            window_size=4,
            num_heads=8,  # 512/8=64 ✓
            use_velocity=use_velocity
        )

        # Channel adjustment to match expected output features
        if img_feature_dims[3] != self.config.perception_output_features:
            self.change_channel_conv_image = nn.Conv2d(
                img_feature_dims[3], self.config.perception_output_features, (1, 1)
            )
            self.change_channel_conv_lidar = nn.Conv2d(
                img_feature_dims[3], self.config.perception_output_features, (1, 1)
            )
        else:
            self.change_channel_conv_image = nn.Sequential()
            self.change_channel_conv_lidar = nn.Sequential()

        # FPN-style top-down pathway
        self._build_fpn_layers()
        
    def _build_fpn_layers(self):
        """Build Feature Pyramid Network layers"""
        channel = self.config.bev_features_chanels
        self.relu = nn.ReLU(inplace=True)
        
        # Top-down pathway
        self.upsample = nn.Upsample(
            scale_factor=self.config.bev_upsample_factor, 
            mode='bilinear', 
            align_corners=False
        )
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        
        # Lateral connections
        self.c5_conv = nn.Conv2d(self.config.perception_output_features, channel, (1, 1))
        
    def top_down(self, x):
        """FPN top-down pathway"""
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample(p4)))
        p2 = self.relu(self.up_conv3(self.upsample(p3)))
        
        return p2, p3, p4, p5

    def forward(self, image, lidar, velocity):
        """
        Forward pass with fixed Swin-PTT fusion
        
        Args:
            image: Input RGB images (B, 3, H, W)
            lidar: Input LiDAR data (B, C, H, W)
            velocity: Ego-vehicle velocity (B, 1)
            
        Returns:
            features: Multi-scale BEV features
            image_features_grid: Image features for auxiliary tasks
            fused_features: Global fused features
        """
        # Normalize images if needed
        if self.image_encoder.normalize:
            image_tensor = normalize_imagenet(image)
        else:
            image_tensor = image

        # ===== ENCODER FORWARD PASSES =====
        
        # Image encoding through CNN backbone
        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.act1(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)
        
        # LiDAR encoding through CNN backbone
        lidar_features = self.lidar_encoder._model.conv1(lidar)
        lidar_features = self.lidar_encoder._model.bn1(lidar_features)
        lidar_features = self.lidar_encoder._model.act1(lidar_features)
        lidar_features = self.lidar_encoder._model.maxpool(lidar_features)

        # ===== MULTI-SCALE SWIN FUSION =====
        
        # Scale 1: Early features
        image_features = self.image_encoder.features.layer1(image_features)
        lidar_features = self.lidar_encoder._model.layer1(lidar_features)
        
        # Apply unified pooling for fusion
        image_embd_1 = self.unified_pool(image_features)
        lidar_embd_1 = self.unified_pool(lidar_features)
        
        img_fused_1, lidar_fused_1 = self.swin_fusion1(image_embd_1, lidar_embd_1, velocity)
        
        # Interpolate back to original sizes and add residual
        img_fused_1 = F.interpolate(img_fused_1, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        lidar_fused_1 = F.interpolate(lidar_fused_1, size=lidar_features.shape[2:], mode='bilinear', align_corners=False)
        image_features = image_features + img_fused_1
        lidar_features = lidar_features + lidar_fused_1

        # Scale 2: Mid-level features
        image_features = self.image_encoder.features.layer2(image_features)
        lidar_features = self.lidar_encoder._model.layer2(lidar_features)
        
        image_embd_2 = self.unified_pool(image_features)
        lidar_embd_2 = self.unified_pool(lidar_features)
        
        img_fused_2, lidar_fused_2 = self.swin_fusion2(image_embd_2, lidar_embd_2, velocity)
        
        img_fused_2 = F.interpolate(img_fused_2, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        lidar_fused_2 = F.interpolate(lidar_fused_2, size=lidar_features.shape[2:], mode='bilinear', align_corners=False)
        image_features = image_features + img_fused_2
        lidar_features = lidar_features + lidar_fused_2

        # Scale 3: Higher-level features
        image_features = self.image_encoder.features.layer3(image_features)
        lidar_features = self.lidar_encoder._model.layer3(lidar_features)
        
        image_embd_3 = self.unified_pool(image_features)
        lidar_embd_3 = self.unified_pool(lidar_features)
        
        img_fused_3, lidar_fused_3 = self.swin_fusion3(image_embd_3, lidar_embd_3, velocity)
        
        img_fused_3 = F.interpolate(img_fused_3, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        lidar_fused_3 = F.interpolate(lidar_fused_3, size=lidar_features.shape[2:], mode='bilinear', align_corners=False)
        image_features = image_features + img_fused_3
        lidar_features = lidar_features + lidar_fused_3

        # Scale 4: Highest-level features
        image_features = self.image_encoder.features.layer4(image_features)
        lidar_features = self.lidar_encoder._model.layer4(lidar_features)
        
        image_embd_4 = self.unified_pool(image_features)
        lidar_embd_4 = self.unified_pool(lidar_features)
        
        img_fused_4, lidar_fused_4 = self.swin_fusion4(image_embd_4, lidar_embd_4, velocity)
        
        img_fused_4 = F.interpolate(img_fused_4, size=image_features.shape[2:], mode='bilinear', align_corners=False)
        lidar_fused_4 = F.interpolate(lidar_fused_4, size=lidar_features.shape[2:], mode='bilinear', align_corners=False)
        image_features = image_features + img_fused_4
        lidar_features = lidar_features + lidar_fused_4

        # ===== FINAL PROCESSING =====
        
        # Adjust channels if needed
        image_features = self.change_channel_conv_image(image_features)
        lidar_features = self.change_channel_conv_lidar(lidar_features)

        # Store grid features for auxiliary tasks
        image_features_grid = image_features
        
        # Global pooling for final features
        image_global = self.image_encoder.features.global_pool(image_features)
        image_global = torch.flatten(image_global, 1)
        
        lidar_global = self.lidar_encoder._model.global_pool(lidar_features)
        lidar_global = torch.flatten(lidar_global, 1)
        
        # Fuse global features
        fused_features = image_global + lidar_global

        # Generate multi-scale features via FPN
        features = self.top_down(lidar_features)
        
        return features, image_features_grid, fused_features


class SwinFusionModule(nn.Module):
    """
    Fixed Swin Transformer-based fusion module with spatial alignment
    """
    
    def __init__(self, img_dim, lidar_dim, fusion_size=(8, 8), window_size=4, num_heads=8, use_velocity=True):
        super().__init__()
        
        self.img_dim = img_dim
        self.lidar_dim = lidar_dim
        self.fusion_size = fusion_size
        self.use_velocity = use_velocity
        
        # ADDED: Validate that dimensions are compatible with num_heads
        assert img_dim % num_heads == 0, f"img_dim ({img_dim}) must be divisible by num_heads ({num_heads})"
        
        # Ensure dimensions match for fusion
        if img_dim != lidar_dim:
            self.dim_align = nn.Conv2d(lidar_dim, img_dim, 1)
        else:
            self.dim_align = nn.Identity()
            
        self.unified_dim = img_dim
        
        # Velocity embedding
        if use_velocity:
            self.vel_emb = nn.Linear(1, self.unified_dim)
        
        # Cross-attention layers for fusion
        self.cross_attn = CrossModalAttention(
            dim=self.unified_dim,
            num_heads=num_heads,
            window_size=window_size
        )
        
        # Self-attention refinement
        self.self_attn_img = SelfAttention(self.unified_dim, num_heads)
        self.self_attn_lidar = SelfAttention(self.unified_dim, num_heads)
        
        # Layer normalization
        self.norm1_img = nn.LayerNorm(self.unified_dim)
        self.norm1_lidar = nn.LayerNorm(self.unified_dim)
        self.norm2_img = nn.LayerNorm(self.unified_dim)
        self.norm2_lidar = nn.LayerNorm(self.unified_dim)
        
        # Feed-forward networks
        self.ffn_img = FeedForward(self.unified_dim)
        self.ffn_lidar = FeedForward(self.unified_dim)
        
    def forward(self, img_features, lidar_features, velocity=None):
        """
        Args:
            img_features: (B, C, H, W)
            lidar_features: (B, C, H, W)  
            velocity: (B, 1)
            
        Returns:
            fused_img_features: (B, C, H, W)
            fused_lidar_features: (B, C, H, W)
        """
        B, C, H, W = img_features.shape
        
        # Align dimensions
        lidar_aligned = self.dim_align(lidar_features)
        
        # Ensure both features have the same spatial size (should be enforced by unified_pool)
        assert img_features.shape == lidar_aligned.shape, f"Spatial mismatch: {img_features.shape} vs {lidar_aligned.shape}"
        
        # Flatten for attention: (B, H*W, C)
        img_tokens = img_features.flatten(2).transpose(1, 2)
        lidar_tokens = lidar_aligned.flatten(2).transpose(1, 2)
        
        # Add velocity embedding if available
        if self.use_velocity and velocity is not None:
            vel_emb = self.vel_emb(velocity)  # (B, C)
            img_tokens = img_tokens + vel_emb.unsqueeze(1)
            lidar_tokens = lidar_tokens + vel_emb.unsqueeze(1)
        
        # Cross-modal attention
        img_cross, lidar_cross = self.cross_attn(img_tokens, lidar_tokens)
        
        # Add residual and norm
        img_tokens = self.norm1_img(img_tokens + img_cross)
        lidar_tokens = self.norm1_lidar(lidar_tokens + lidar_cross)
        
        # Self-attention refinement
        img_self = self.self_attn_img(img_tokens)
        lidar_self = self.self_attn_lidar(lidar_tokens)
        
        img_tokens = self.norm2_img(img_tokens + img_self)
        lidar_tokens = self.norm2_lidar(lidar_tokens + lidar_self)
        
        # Feed-forward
        img_tokens = img_tokens + self.ffn_img(img_tokens)
        lidar_tokens = lidar_tokens + self.ffn_lidar(lidar_tokens)
        
        # Reshape back to spatial format
        fused_img = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        fused_lidar = lidar_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        return fused_img, fused_lidar


class CrossModalAttention(nn.Module):
    """Cross-modal attention between image and LiDAR features"""
    
    def __init__(self, dim, num_heads=8, window_size=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        # ADDED: Validation
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        # Query, Key, Value projections
        self.q_img = nn.Linear(dim, dim)
        self.k_lidar = nn.Linear(dim, dim)
        self.v_lidar = nn.Linear(dim, dim)
        
        self.q_lidar = nn.Linear(dim, dim)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        
        self.proj_img = nn.Linear(dim, dim)
        self.proj_lidar = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)
        
    def forward(self, img_tokens, lidar_tokens):
        B, N, C = img_tokens.shape
        
        # REMOVED: Debug prints that were causing issues
        # Image to LiDAR attention
        q_img = self.q_img(img_tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_lidar = self.k_lidar(lidar_tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_lidar = self.v_lidar(lidar_tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn_img_to_lidar = (q_img @ k_lidar.transpose(-2, -1)) * self.scale
        attn_img_to_lidar = F.softmax(attn_img_to_lidar, dim=-1)
        attn_img_to_lidar = self.attn_drop(attn_img_to_lidar)
        
        img_enhanced = (attn_img_to_lidar @ v_lidar).transpose(1, 2).reshape(B, N, C)
        img_enhanced = self.proj_img(img_enhanced)
        img_enhanced = self.proj_drop(img_enhanced)
        
        # LiDAR to Image attention
        q_lidar = self.q_lidar(lidar_tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k_img = self.k_img(img_tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_img = self.v_img(img_tokens).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn_lidar_to_img = (q_lidar @ k_img.transpose(-2, -1)) * self.scale
        attn_lidar_to_img = F.softmax(attn_lidar_to_img, dim=-1)
        attn_lidar_to_img = self.attn_drop(attn_lidar_to_img)
        
        lidar_enhanced = (attn_lidar_to_img @ v_img).transpose(1, 2).reshape(B, N, C)
        lidar_enhanced = self.proj_lidar(lidar_enhanced)
        lidar_enhanced = self.proj_drop(lidar_enhanced)
        
        return img_enhanced, lidar_enhanced


class SelfAttention(nn.Module):
    """Standard self-attention module"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # ADDED: Validation
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


# ===== Auxiliary Components =====

class ImageCNN(nn.Module):
    """CNN encoder for image input with TIMM models"""

    def __init__(self, architecture, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = None
        
        # Handle different architectures
        if architecture.startswith('regnet'):
            self.features.conv1 = self.features.stem.conv
            self.features.bn1 = self.features.stem.bn
            self.features.act1 = nn.Sequential()
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.s1
            self.features.layer2 = self.features.s2
            self.features.layer3 = self.features.s3
            self.features.layer4 = self.features.s4
            self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.features.head = nn.Sequential()

        elif architecture.startswith('convnext'):
            self.features.conv1 = self.features.stem._modules['0']
            self.features.bn1 = self.features.stem._modules['1']
            self.features.act1 = nn.Sequential()
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.stages._modules['0']
            self.features.layer2 = self.features.stages._modules['1']
            self.features.layer3 = self.features.stages._modules['2']
            self.features.layer4 = self.features.stages._modules['3']
            self.features.global_pool = self.features.head
            self.features.global_pool.flatten = nn.Sequential()
            self.features.global_pool.fc = nn.Sequential()
            self.features.head = nn.Sequential()


class LidarEncoder(nn.Module):
    """Standard LiDAR encoder compatible with TransFuser"""
    
    def __init__(self, architecture, in_channels=2):
        super().__init__()

        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        if architecture.startswith('regnet'):
            self._model.conv1 = self._model.stem.conv
            self._model.bn1  = self._model.stem.bn
            self._model.act1 = nn.Sequential()
            self._model.maxpool =  nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()

        elif architecture.startswith('convnext'):
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules['0']
            self._model.layer2 = self._model.stages._modules['1']
            self._model.layer3 = self._model.stages._modules['2']
            self._model.layer4 = self._model.stages._modules['3']
            self._model.global_pool = self._model.head
            self._model.global_pool.flatten = nn.Sequential()
            self._model.global_pool.fc = nn.Sequential()
            self._model.head = nn.Sequential()

        # Change first conv layer for LiDAR channels
        _tmp = self._model.conv1
        use_bias = (_tmp.bias is not None)
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=use_bias)
        
        # Clean up
        if architecture.startswith('convnext'):
            del self._model.stem._modules['0']
        elif architecture.startswith('regnet'):
            del self._model.stem.conv
            
        if use_bias:
            self._model.conv1.bias = _tmp.bias
        del _tmp


def normalize_imagenet(x):
    """Normalize input images according to ImageNet standards"""
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


# ===== Decoder Components =====

class SegDecoder(nn.Module):
    """Segmentation decoder for auxiliary tasks"""
    
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.num_class = config.num_class

        self.deconv1 = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
            nn.ReLU(True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_3, self.num_class, 3, 1, 1),
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)
        return x


class DepthDecoder(nn.Module):
    """Depth estimation decoder for auxiliary tasks"""
    
    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim

        self.deconv1 = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.config.deconv_channel_num_1, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_1, self.config.deconv_channel_num_2, 3, 1, 1),
            nn.ReLU(True),
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(self.config.deconv_channel_num_2, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(self.config.deconv_channel_num_3, self.config.deconv_channel_num_3, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.config.deconv_channel_num_3, 1, 3, 1, 1),
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.config.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)
        x = torch.sigmoid(x).squeeze(1)
        return x
