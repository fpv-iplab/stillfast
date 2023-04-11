from stillfast.models.faster_rcnn import FasterRCNN
from stillfast.models.faster_rcnn_sta import RoiHeadsSTA, RoiHeadsSTAv2
from .build import MODEL_REGISTRY
from detectron2.config import configurable
from stillfast.models.backbone_utils_stillfast import StillFastBackbone, StillBackbone
from stillfast.transforms import StillFastTransform
from torchvision.models.detection._utils import overwrite_eps
from torchvision._internally_replaced_utils import load_state_dict_from_url
from stillfast.datasets import StillFastImageTensor


@MODEL_REGISTRY.register()
class StillFast(FasterRCNN):
    @configurable
    def __init__(self, *args, pretrained, transform, **kwargs) -> None:
        super().__init__(*args, pretrained=False, **kwargs)

        if pretrained:
            self.load_faster_rcnn_pretrained_weights(kwargs)

        self.transform = transform

    def load_faster_rcnn_pretrained_weights(self, kwargs):
        print(f"Loading checkpoint from {kwargs['checkpoint_url']}")
        # Load state dict
        state_dict = load_state_dict_from_url(kwargs['checkpoint_url'], progress=True)

        # load pretrained weights for backbone
        missing_keys, unmatched_keys = self.backbone.load_faster_rcnn_pretrained_weights(state_dict)

        # Discard keys that have already been matched
        state_dict = {k: v for k, v in state_dict.items() if k in unmatched_keys}

        # load head only if not replaced
        if not kwargs['replace_head']:
            head_state_dict = {k.replace('roi_heads.',''): v for k, v in state_dict.items() if 'roi_heads' in k}
            m,u = self.roi_heads.load_state_dict(head_state_dict)
            missing_keys += m
            unmatched_keys += u
        else:
            # ignore head weights
            unmatched_keys = [k for k in unmatched_keys if 'roi_heads' not in k]
            print("Skipping roi_heads weights as the head has been replaced")

        # load rpn weights
        rpn_state_dict = {k.replace('rpn.',''):v for k,v in state_dict.items() if 'rpn' in k}
        m, unmatched_keys = self.rpn.load_state_dict(rpn_state_dict, strict=False)

        missing_keys += m
        
        print(f"Missing keys: {missing_keys}")
        print(f"Unmatched keys: {unmatched_keys}")
        
        overwrite_eps(self, 0.0)

    @classmethod
    def from_config(cls, cfg):
        options = super().from_config(cfg)
        if (cfg.MODEL.BRANCH == 'Still'):
            backbone = StillBackbone(cfg.MODEL)
        else: 
            backbone = StillFastBackbone(cfg.MODEL)
        transform = StillFastTransform(cfg)

        hver = cfg.MODEL.STILLFAST.ROI_HEADS.VERSION

        if  hver == 'v1':
            roi_heads = RoiHeadsSTA(cfg)
        elif hver == 'v2':
            roi_heads = RoiHeadsSTAv2(cfg)
        else:
            raise ValueError(f"Unknown version of RoiHeads: {hver}")

        options.update({
            'roi_heads': roi_heads,
            'backbone': backbone,
            'transform': transform
        })
            
        return options 

    def forward(self, batch):
        images = [StillFastImageTensor(a, b) for a,b in zip(batch['still_img'], batch['fast_imgs'])]
        targets = batch['targets'] if 'targets' in batch else None
        return super().forward(images, targets)
    