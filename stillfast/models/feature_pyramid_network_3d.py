from torch import nn
from typing import List, Optional, Dict
from torch import Tensor
from torch.nn import functional as F
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock

class FeaturePyramidNetwork3D(FeaturePyramidNetwork):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super().__init__(in_channels_list, out_channels, extra_blocks)
        
        # Replace conv layers
        for i in range(len(self.inner_blocks)):
            ib = self.inner_blocks[i]
            lb = self.layer_blocks[i]
            self.inner_blocks[i] = nn.Conv3d(ib.in_channels, ib.out_channels, 1)
            self.layer_blocks[i] = nn.Conv3d(lb.out_channels, lb.out_channels, 3, padding=1)
            
        # Init Conv3d
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-3:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # pack it back into an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool3D(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x, y, names):
        names.append("pool")
        x.append(F.max_pool3d(x[-1], 1, (1,2,2), 0))
        return x, names