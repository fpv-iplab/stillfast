from typing import Optional, Callable
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, resnet_fpn_backbone, BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from torch import nn
from torchvision.models import resnet

def build_still_backbone(
    backbone_name: str,
    trainable_layers: int,
    pretrained: bool
) -> BackboneWithFPN:
    trainable_layers = _validate_trainable_layers(
        pretrained, trainable_layers, 5, 3
    )

    return resnet_fpn_backbone(backbone_name, pretrained, trainable_layers = trainable_layers)

def build_clean_2d_backbone(
    backbone_name: str, 
    pretrained: bool,
    trainable_layers: int,
    norm_layer: Optional[Callable[..., nn.Module]] = misc_nn_ops.FrozenBatchNorm2d
) -> nn.Module:
    if backbone_name not in ['resnet50']:
        raise ValueError(f"Backbone {backbone_name} is not supported with 3D models")
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=norm_layer)
    backbone.channels = [256, 512, 1024, 2048]
    del backbone.avgpool
    del backbone.fc

    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return backbone