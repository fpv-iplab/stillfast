import functools
from imaplib import Int2AP
import torch
from torch import nn
from detectron2.config import configurable
from torch.nn import functional as F
from typing import List
from stillfast.models.backbone_utils_2d import build_clean_2d_backbone, build_still_backbone
from stillfast.models.backbone_utils_3d import build_clean_3d_backbone
from functools import partial
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork
from stillfast.models.backbone_utils_3d import FeaturePyramidNetwork3D, LastLevelMaxPool3D
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from stillfast.ops.misc import Conv2dNormActivation
from typing import Union
import math

class NonLocalTemporalPooling(nn.Module):
    def __init__(self, num_channels, inter_channels, max_height_before_pooling):
        super().__init__()
        if inter_channels is None or inter_channels == 'half':
            self.inter_channels = num_channels // 2
            
        if inter_channels==0:
            self.inter_channels = 1
            
        self.num_channels = num_channels

        self.max_height = max_height_before_pooling
            
        # All 1x1 filters
        self.q = nn.Conv2d(self.num_channels, self.inter_channels, 1)
        self.k = nn.Conv3d(self.num_channels, self.inter_channels, 1)
        self.v = nn.Conv3d(self.num_channels, self.inter_channels, 1)
        
        self.out_conv = nn.Conv2d(self.inter_channels, num_channels, 1)
        
        # Initialize to zero so that the module implements an identity function at initialization
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)
        
    def forward(self, x):
        BS, _, T, H, W = x.shape
        NC = self.inter_channels
        last_frame = x[:,:,-1,:,:]
        Q = self.q(last_frame) 
        if H>self.max_height:
            k = math.ceil(H/self.max_height)
            x_pool = F.max_pool3d(x, kernel_size=(1, k, k))
        else:
            x_pool = x
        K = self.k(x_pool)
        V = self.v(x_pool)
        
        #concat H and W and re-arrange to have BS x HW x NC
        Q = Q.view(BS, NC, -1).permute(0,2,1)
        #concat H and W and T 
        K = K.view(BS, NC, -1)
        #concat H and W and T and re-arrange to have BS x HWT x NC
        V = V.view(BS, NC, -1).permute(0,2,1)
        
        att = F.softmax(torch.matmul(Q, K), dim=-1)
        att = F.softmax(att, dim=-1)
        
        out = torch.matmul(att, V) #BS x HW x NC
        #rearrange and reshape to obtain BS x NC x H x W
        out = out.permute(0, 2, 1).view(BS, NC, H, W)
        
        # residual connection
        return last_frame + self.out_conv(out)

class NonLocalFusionBlock(nn.Module):
    @configurable
    def __init__(self, 
                 channels_2d,
                 channels_3d,
                 inter_channels,
                 max_height_before_scaling_2d,
                 max_height_before_pooling_3d,
                 post_sum_conv,
                 scaling_2d_mode):
        super().__init__()
        self.channels_2d = channels_2d
        self.channels_3d = channels_3d

        self.max_height_3d = max_height_before_pooling_3d
        self.max_height_2d = max_height_before_scaling_2d
        
        self.scaling_2d_mode = scaling_2d_mode
        
        if inter_channels is None or inter_channels == 'half':
            self.inter_channels = channels_2d // 2
            
        if inter_channels==0:
            self.inter_channels = 1
            
        # All 1x1 filters
        self.q = nn.Conv2d(self.channels_2d, self.inter_channels, 1)
        self.k = nn.Conv3d(self.channels_3d, self.inter_channels, 1)
        self.v = nn.Conv3d(self.channels_3d, self.inter_channels, 1)
        
        self.out_conv = nn.Conv2d(self.inter_channels, self.channels_2d, 1)
        
        # Initialize to zero so that the module implements an identity function at initialization
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)

        self.post_sum_conv = post_sum_conv
        if post_sum_conv:
            self.post_sum_conv = nn.Conv2d(self.channels_2d, self.channels_2d, kernel_size=(3,3), padding=(1,1))
        else:
            self.post_sum_conv = nn.Identity()

    @classmethod
    def from_config(cls, cfg):
        return {
            'inter_channels': cfg.INTER_CHANNELS,
            'max_height_before_scaling_2d': cfg.MAX_HEIGHT_BEFORE_SCALING_2D,
            'max_height_before_pooling_3d': cfg.MAX_HEIGHT_BEFORE_POOLING_3D,
            'post_sum_conv': cfg.POST_SUM_CONV_BLOCK,
            'scaling_2d_mode': cfg.SCALING_2D_MODE
        }
        
    def forward(self, features_2d, features_3d):
        BS, _, H, W = features_2d.shape
        if H>self.max_height_2d:
            scale_factor = self.max_height_2d/H
            features_2d_scaled = F.interpolate(features_2d, scale_factor=scale_factor, mode=self.scaling_2d_mode, recompute_scale_factor=True)
            _, _, H, W = features_2d_scaled.shape
        else:
            features_2d_scaled = features_2d
        _, _, _, H3d, _ = features_3d.shape
        NC = self.inter_channels
        Q = self.q(features_2d_scaled) 
        if H3d>self.max_height_3d:
            k = math.ceil(H3d/self.max_height_3d)
            features_3d_pool = F.max_pool3d(features_3d, kernel_size=(1, k, k))
        else:
            features_3d_pool = features_3d
        K = self.k(features_3d_pool)
        V = self.v(features_3d_pool)
        
        #concat H and W and re-arrange to have BS x HW x NC
        Q = Q.view(BS, NC, -1).permute(0,2,1)
        #concat H and W and T 
        K = K.view(BS, NC, -1)
        #concat H and W and T and re-arrange to have BS x HWT x NC
        V = V.view(BS, NC, -1).permute(0,2,1)
        
        att = torch.matmul(Q, K)
        att = F.softmax(att, dim=-1)
        
        out = torch.matmul(att, V) #BS x HW x NC
        #rearrange and reshape to obtain BS x NC x H x W
        out = out.permute(0, 2, 1).view(BS, NC, H, W)

        out = self.out_conv(out)

        if features_2d_scaled.shape != features_2d.shape:
            out = F.interpolate(out, size=features_2d.shape[2:], mode=self.scaling_2d_mode)

        out = self.post_sum_conv(features_2d + out)
        
        # residual connection
        return out
class ConvolutionalFusionBlock(nn.Module):
    @configurable
    def __init__(
        self, 
        pooling: str,
        conv_block_architecture: str,
        post_up_conv_block: bool,
        post_sum_conv_block: bool,
        gating_block: str,
        channels_2d: int,
        channels_3d: int,
        pooling_frames: int,
        temporal_nonlocal_pooling_inter_channels: Union[str,int],
        temporal_nonlocal_pooling_max_height_before_max_pooling: int
    ):
        super().__init__()

        self.pooling = pooling
        self.conv_block_architecture = conv_block_architecture

        if self.pooling == 'conv':
            self.conv_pooling = nn.Conv3d(channels_3d, channels_3d, kernel_size=(pooling_frames,1,1), padding=(0,0,0))
        elif self.pooling == 'channel_stack':
            self.channel_stack_pooling = nn.Conv2d(channels_3d*pooling_frames, channels_3d, kernel_size=1)
        elif self.pooling == 'nonlocal':
            self.nonlocal_pooling = NonLocalTemporalPooling(channels_3d, temporal_nonlocal_pooling_inter_channels, temporal_nonlocal_pooling_max_height_before_max_pooling)

        if post_up_conv_block:
            self.post_up_conv_block = self._build_conv_block(channels_3d, channels_2d)
        else:
            self.post_up_conv_block = nn.Identity()

        if post_sum_conv_block:
            self.post_sum_conv_block = self._build_conv_block(channels_2d, channels_2d)
        else:
            self.post_sum_conv_block = nn.Identity()

        if gating_block == 'channel':
            self.gating_block = nn.Sequential(
                    nn.Linear(channels_2d+channels_3d, channels_2d),
                    nn.ReLU(True),
                    nn.Linear(channels_2d, channels_2d),
                    nn.Sigmoid()
                )
        elif gating_block == None or gating_block == 'None':
            self.gating_block = None
        else:
            raise ValueError(f'Unknown gating block: {gating_block}')

    def _build_conv_block(self, in_channels, out_channels):
        if self.conv_block_architecture == 'simple_convolution':
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        elif self.conv_block_architecture == 'Conv2dNormActivation':
            return Conv2dNormActivation(in_channels, out_channels, kernel_size=3)
        else:
            raise ValueError(f'Unknown convolution block architecture: {self.conv_block_architecture}')

    @classmethod
    def from_config(cls, cfg):
        return {
            'pooling': cfg.POOLING,
            'conv_block_architecture': cfg.CONV_BLOCK_ARCHITECTURE,
            'post_up_conv_block': cfg.POST_UP_CONV_BLOCK,
            'post_sum_conv_block': cfg.POST_SUM_CONV_BLOCK,
            'gating_block': cfg.GATING_BLOCK,
            'pooling_frames': cfg.POOLING_FRAMES,
            'temporal_nonlocal_pooling_inter_channels': cfg.TEMPORAL_NONLOCAL_POOLING.INTER_CHANNELS,
            'temporal_nonlocal_pooling_max_height_before_max_pooling': cfg.TEMPORAL_NONLOCAL_POOLING.MAX_HEIGHT_BEFORE_POOLING
        }

    def _pool(self, x):
        if self.pooling == 'max':
            return torch.max(x, dim=2)[0]
        elif self.pooling == 'mean':
            return torch.mean(x, dim=2)
        elif self.pooling == 'last':
            return x[:, :, -1, :]
        elif self.pooling == 'conv':
            return self.conv_pooling(x).squeeze(2)
        elif self.pooling == 'channel_stack':
            return self.channel_stack_pooling(x.view(x.shape[0],-1,x.shape[3], x.shape[4]))
        elif self.pooling == 'nonlocal':
            return self.nonlocal_pooling(x)
        else:
            raise ValueError(f'Unknown pooling: {self.pooling}')

    def forward(self, in_2d, in_3d):
        pooled_3d = self._pool(in_3d)
        up_3d = F.interpolate(pooled_3d, in_2d.shape[-2:], mode="nearest")
        up_3d = self.post_up_conv_block(up_3d)

        if self.gating_block is not None:
            p2d=in_2d.view(*in_2d.shape[:2], -1).mean(-1)
            p3d=in_3d.view(*in_3d.shape[:2], -1).mean(-1)

            gating_values = self.gating_block(torch.cat([p2d, p3d], dim=-1))
            gating_values = gating_values.view(up_3d.shape[0], up_3d.shape[1], 1, 1)
            up_3d = up_3d * gating_values

        fuse_2d = in_2d + up_3d
        fuse_2d = self.post_sum_conv_block(fuse_2d)

        return fuse_2d
            
        
class StillFastBackbone(nn.Module):
    @configurable
    def __init__(
        self,
        still_backbone: nn.Module,
        fast_backbone: nn.Module,
        still_backbone_channels: List[int],
        fast_backbone_channels: List[int],
        pre_pyramid_fusion: bool,
        post_pyramid_fusion: bool,
        lateral_connections: bool,
        fusion_block,
        layers: List[int] = range(1, 5),
        pyramid_channels: int = 256,
        ) -> None:
        super().__init__()

        self.still_backbone = still_backbone
        self.fast_backbone = fast_backbone

        self.still_backbone_channels = still_backbone_channels
        self.fast_backbone_channels = fast_backbone_channels

        self.pre_pyramid_fusion = pre_pyramid_fusion
        self.post_pyramid_fusion = post_pyramid_fusion
        self.lateral_connections = lateral_connections

        self.layers = layers

        if pre_pyramid_fusion:
            for layer, c2d, c3d in zip(layers, still_backbone_channels, fast_backbone_channels):
                setattr(self, f"pre_pyramid_fusion_block_{layer}", fusion_block(channels_2d=c2d, channels_3d=c3d))

        if post_pyramid_fusion:
            for layer, c2d, c3d in zip(layers, still_backbone_channels, fast_backbone_channels):
                setattr(self, f"post_pyramid_fusion_block_{layer}", fusion_block(channels_2d=pyramid_channels, channels_3d=pyramid_channels))

        if lateral_connections:
            for layer, c2d, c3d in zip(layers, still_backbone_channels, fast_backbone_channels):
                setattr(self, f"lateral_connection_fusion_block_{layer}", fusion_block(channels_2d=c2d, channels_3d=c3d))

        self.still_fpn = FeaturePyramidNetwork(still_backbone_channels, pyramid_channels, extra_blocks=LastLevelMaxPool())
        
        if post_pyramid_fusion:
            self.fast_fpn = FeaturePyramidNetwork3D(fast_backbone_channels, pyramid_channels, extra_blocks=LastLevelMaxPool3D())

        self.out_channels = pyramid_channels

    @classmethod
    def from_config(cls, cfg):
        still_backbone = build_clean_2d_backbone(
            backbone_name = cfg.STILL.BACKBONE.NAME,
            pretrained = cfg.STILL.BACKBONE.PRETRAINED,
            trainable_layers=cfg.STILL.BACKBONE.TRAINABLE_LAYERS
            )
        
        fast_backbone = build_clean_3d_backbone(
            backbone_name = cfg.FAST.BACKBONE.NAME,
            pretrained = cfg.FAST.BACKBONE.PRETRAINED,
            temporal_causal_conv3d= cfg.FAST.BACKBONE.TEMPORAL_CAUSAL_CONV3D
        )

        if cfg.STILLFAST.FUSION.FUSION_BLOCK == 'convolutional':
            fusion_block = partial(ConvolutionalFusionBlock, cfg.STILLFAST.FUSION.CONVOLUTIONAL_FUSION_BLOCK)
        elif cfg.STILLFAST.FUSION.FUSION_BLOCK == 'nonlocal':
            fusion_block = partial(NonLocalFusionBlock, cfg.STILLFAST.FUSION.NONLOCAL_FUSION_BLOCK)
        else:
            raise ValueError(f'Unknown fusion block: {cfg.STILLFAST.FUSION.FUSION_BLOCK}')

        return {
            'still_backbone': still_backbone,
            'fast_backbone': fast_backbone,
            'still_backbone_channels': still_backbone.channels,
            'fast_backbone_channels': fast_backbone.channels,
            'pre_pyramid_fusion': cfg.STILLFAST.FUSION.PRE_PYRAMID_FUSION,
            'post_pyramid_fusion': cfg.STILLFAST.FUSION.POST_PYRAMID_FUSION,
            'lateral_connections': cfg.STILLFAST.FUSION.LATERAL_CONNECTIONS,
            'fusion_block': fusion_block
        }

    def load_faster_rcnn_pretrained_weights(self, state_dict):
        # Loads faster rcnn pretrained backbone weights into the still backbone
        # These include the ResNet backbone and the FPN
        # ROI HEADS and other layers are not loaded here

        # Load backbone and keep track of missing keys (should be zero) and unmatched keys
        state_dict = {k.replace('backbone.body.',''):v for k,v in state_dict.items()}
        missing_keys, unmatched_keys = self.still_backbone.load_state_dict(state_dict, strict=False)

        # Remove keys that have been aldreay loaded and remap
        state_dict = {k.replace('backbone.fpn.',''): v for k , v in state_dict.items() if k in unmatched_keys}
        # Load FPN and keep track of missing keys (should be zero) and unmatched keys
        m, unmatched_keys = self.still_fpn.load_state_dict(state_dict, strict=False)

        missing_keys += m #Add missing keys (should be zero)
        return missing_keys, unmatched_keys #return missing and unmatched keys

    def forward(self, x):
        h_still, h_fast = x
        
        # Basic Stem
        h_still = self.still_backbone.conv1(h_still)
        h_still = self.still_backbone.bn1(h_still)
        h_still = self.still_backbone.relu(h_still)
        h_still = self.still_backbone.maxpool(h_still)
        
        # Basic Stem
        h_fast = self.fast_backbone.blocks[0](h_fast)
        
        still_features = OrderedDict()
        fast_features = OrderedDict()

        # Forward through the backbones, layer by layer
        for layer in self.layers:
            layer_still = getattr(self.still_backbone, f"layer{layer}")
            layer_fast = self.fast_backbone.blocks[layer]
            
            h_still = layer_still(h_still)
            h_fast = layer_fast(h_fast)

            # If lateral connections are enabled
            # update h_still before going to the next layer
            if self.lateral_connections:
                h_still = getattr(self, f"lateral_connection_fusion_block_{layer}") (h_still, h_fast)
            
            still_features[f"{layer-1}"] = h_still
            fast_features[f"{layer-1}"] = h_fast

        keys = list(fast_features.keys())
        self.memory3d = fast_features[keys[-1]]

        # If pre-pyramid fusion is enabled
        # fuse all fast maps into still maps
        # this is done after computing all still features
        # so it does not implement lateral connections
        if self.pre_pyramid_fusion:
            for layer in self.layers:
                still_features[f"{layer-1}"] = getattr(self, f"pre_pyramid_fusion_block_{layer}") (still_features[f"{layer-1}"], fast_features[f"{layer-1}"])
        
        # Apply the feature pyramid to the still features
        out_features = self.still_fpn(still_features)

        # If post-pyramid fusion is enabled
        if self.post_pyramid_fusion:
            # Apply the feature pyramid to the fast features
            fast_features = self.fast_fpn(fast_features)
            # Fuse fast features into still features
            for layer in self.layers:
                out_features[f"{layer-1}"] = getattr(self, f"post_pyramid_fusion_block_{layer}") (out_features[f"{layer-1}"], fast_features[f"{layer-1}"])

        return out_features


class StillBackbone(nn.Module):
    # implement only the backbone which processes a still image
    @configurable
    def __init__(
        self,
        still_backbone: nn.Module,
        still_backbone_channels: List[int],
        layers: List[int] = range(1, 5),
        pyramid_channels: int = 256,
        ) -> None:
        super().__init__()

        self.still_backbone = still_backbone

        self.still_backbone_channels = still_backbone_channels

        self.layers = layers

        self.still_fpn = FeaturePyramidNetwork(still_backbone_channels, pyramid_channels, extra_blocks=LastLevelMaxPool())
        

        self.out_channels = pyramid_channels

    @classmethod
    def from_config(cls, cfg):
        still_backbone = build_clean_2d_backbone(
            backbone_name = cfg.STILL.BACKBONE.NAME,
            pretrained = cfg.STILL.BACKBONE.PRETRAINED,
            trainable_layers=cfg.STILL.BACKBONE.TRAINABLE_LAYERS
            )
        
        return {
            'still_backbone': still_backbone,
            'still_backbone_channels': still_backbone.channels
        }

    def load_faster_rcnn_pretrained_weights(self, state_dict):
        # Loads faster rcnn pretrained backbone weights into the still backbone
        # These include the ResNet backbone and the FPN
        # ROI HEADS and other layers are not loaded here

        # Load backbone and keep track of missing keys (should be zero) and unmatched keys
        state_dict = {k.replace('backbone.body.',''):v for k,v in state_dict.items()}
        missing_keys, unmatched_keys = self.still_backbone.load_state_dict(state_dict, strict=False)

        # Remove keys that have been aldreay loaded and remap
        state_dict = {k.replace('backbone.fpn.',''): v for k , v in state_dict.items() if k in unmatched_keys}
        # Load FPN and keep track of missing keys (should be zero) and unmatched keys
        m, unmatched_keys = self.still_fpn.load_state_dict(state_dict, strict=False)

        missing_keys += m #Add missing keys (should be zero)
        return missing_keys, unmatched_keys #return missing and unmatched keys

    def forward(self, x):
        h_still, h_fast = x
        
        # Basic Stem
        h_still = self.still_backbone.conv1(h_still)
        h_still = self.still_backbone.bn1(h_still)
        h_still = self.still_backbone.relu(h_still)
        h_still = self.still_backbone.maxpool(h_still)
        
        # Basic Stem        
        still_features = OrderedDict()

        # Forward through the backbones, layer by layer
        for layer in self.layers:
            layer_still = getattr(self.still_backbone, f"layer{layer}")
            h_still = layer_still(h_still)
            still_features[f"{layer-1}"] = h_still
           
        # Apply the feature pyramid to the still features
        out_features = self.still_fpn(still_features)

        return out_features

