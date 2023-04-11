from typing import Optional, List, Dict
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from stillfast.models.feature_pyramid_network_3d import FeaturePyramidNetwork3D, LastLevelMaxPool3D
from torch import nn
from torch.nn import functional as F
import torch
from torch import Tensor
from stillfast.ops.misc import TemporalCausalConv3D

def replace_module(m, replace_type, replace_func):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == replace_type:
            setattr(m, attr_str, TemporalCausalConv3D.from_conv3d(target_attr))
    for n, ch in m.named_children():
        replace_module(ch, replace_type, replace_func)

def build_clean_3d_backbone(
    backbone_name: str, 
    pretrained: bool,
    temporal_causal_conv3d: bool = False,
):
    if backbone_name not in ['slow_r50', 'x3d_l', 'x3d_m', 'r2plus1d_r50']:
        raise ValueError(f"Backbone {backbone_name} is not supported with 3D models")
    backbone = torch.hub.load("facebookresearch/pytorchvideo", model=backbone_name, pretrained=pretrained)
    if backbone_name in ['slow_r50', 'r2plus1d_r50']:
        channels_list = [256, 512, 1024, 2048]
    elif backbone_name in ['x3d_l', 'x3d_m']:
        channels_list = [24, 48, 96, 192]
    backbone.channels = channels_list
    del backbone.blocks[5]

    if temporal_causal_conv3d:
        replace_module(backbone, nn.Conv3d, TemporalCausalConv3D.from_conv3d)
        print('Replaced all Conv3d in the fast backbone with TemporalCausalConv3D keeping the same weights')

    return backbone