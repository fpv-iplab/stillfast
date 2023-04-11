from turtle import forward
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection._utils import overwrite_eps
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection.faster_rcnn import model_urls
from .build import MODEL_REGISTRY
import torch
from stillfast.transforms import GeneralizedRCNNTransformWithHorizontalFlip
from detectron2.config import configurable
from torchvision.models.detection.roi_heads import RoIHeads as pytorch_ROIHeads
from torchvision.ops import MultiScaleRoIAlign
from stillfast.models.backbone_utils_2d import build_still_backbone

class RoiHeads(pytorch_ROIHeads):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.STILL.BACKBONE.NAME == 'resnet50':
            out_channels=256
        else:
            raise NotImplementedError

        representation_size = cfg.MODEL.STILL.BOX.PREDICTOR_REPRESENTATION_SIZE
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, 
                        sampling_ratio=cfg.MODEL.STILL.BOX.POOLER_SAMPLING_RATIO)
        resolution = box_roi_pool.output_size[0]
        box_head = faster_rcnn.TwoMLPHead(out_channels * resolution ** 2, representation_size)
        # Using 91 as num_classes to allow loading pretrained weights
        # This will be changed later when the ROI HEAD is replaced
        box_predictor = faster_rcnn.FastRCNNPredictor(representation_size, 91)

        return {
            'box_roi_pool': box_roi_pool,
            'box_head': box_head,
            'box_predictor': box_predictor,
            'fg_iou_thresh': cfg.MODEL.STILL.BOX.FG_IOU_THRESH,
            'bg_iou_thresh': cfg.MODEL.STILL.BOX.BG_IOU_THRESH,
            'batch_size_per_image': cfg.MODEL.STILL.BOX.BATCH_SIZE_PER_IMAGE,
            'positive_fraction': cfg.MODEL.STILL.BOX.POSITIVE_FRACTION,
            'bbox_reg_weights': cfg.MODEL.STILL.BOX.REG_WEIGHTS,
            'score_thresh': cfg.MODEL.STILL.BOX.SCORE_THRESH,
            'nms_thresh': cfg.MODEL.STILL.BOX.NMS_THRESH,
            'detections_per_img': cfg.MODEL.STILL.BOX.DETECTIONS_PER_IMG
        }
@MODEL_REGISTRY.register()
class FasterRCNN(faster_rcnn.FasterRCNN):
    @configurable
    def __init__(
        self,
        backbone,
        pretrained,
        checkpoint_url,
        num_classes,
        replace_head,
        train_random_flip,
        min_size,
        max_size,
        image_mean,
        image_std,
        roi_heads,
        **kwargs
    ):
        super().__init__(
            backbone, 
            num_classes = num_classes, 
            min_size=min_size, 
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs
            )

        self.roi_heads = roi_heads

        if pretrained:
            print(f"Loading checkpoint from {checkpoint_url}")
            state_dict = load_state_dict_from_url(checkpoint_url, progress=True)
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            missing_keys = [k for k in missing_keys if 'verb_predictor' not in k and 'ttc_predictor' not in k]
            print('Ignoring verb_predictor and ttc_predictor missing keys (these are not supposed to be in the checkpoint)')
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            overwrite_eps(self, 0.0)
        
        if replace_head:
            in_features = self.roi_heads.box_predictor.cls_score.in_features
            self.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        else:
            assert num_classes == 91, "replace_head must be true to change num_classes"
        
        self.transform = GeneralizedRCNNTransformWithHorizontalFlip(
                min_size = min_size,
                max_size = max_size,
                image_mean = image_mean,
                image_std = image_std,
                train_horizontal_flip=train_random_flip
            )

    @classmethod
    def from_config(cls, cfg):
        backbone = build_still_backbone(
            cfg.MODEL.STILL.BACKBONE.NAME, 
            cfg.MODEL.STILL.BACKBONE.TRAINABLE_LAYERS,
            cfg.MODEL.STILL.BACKBONE.PRETRAINED
        )

        url = [m for m in model_urls if cfg.MODEL.STILL.BACKBONE.NAME in m][0]
        checkpoint_url = model_urls[url]

        return {
            'backbone': backbone,
            'pretrained': cfg.MODEL.STILL.PRETRAINED,
            'checkpoint_url': checkpoint_url,
            'num_classes': cfg.MODEL.NOUN_CLASSES+1 if cfg.MODEL.STILL.REPLACE_HEAD else 91,
            'min_size': cfg.DATA.STILL.MIN_SIZE,
            'max_size': cfg.DATA.STILL.MAX_SIZE,
            'image_mean': cfg.DATA.STILL.MEAN,
            'image_std': cfg.DATA.STILL.STD,
            'rpn_anchor_generator': cfg.MODEL.STILL.RPN.ANCHOR_GENERATOR,
            'rpn_head': cfg.MODEL.STILL.RPN.HEAD,
            'rpn_pre_nms_top_n_train': cfg.MODEL.STILL.RPN.PRE_NMS_TOP_N_TRAIN,
            'rpn_pre_nms_top_n_test': cfg.MODEL.STILL.RPN.PRE_NMS_TOP_N_TEST,
            'rpn_post_nms_top_n_train': cfg.MODEL.STILL.RPN.POST_NMS_TOP_N_TRAIN,
            'rpn_post_nms_top_n_test': cfg.MODEL.STILL.RPN.POST_NMS_TOP_N_TEST,
            'rpn_nms_thresh': cfg.MODEL.STILL.RPN.NMS_THRESH,
            'rpn_fg_iou_thresh': cfg.MODEL.STILL.RPN.FG_IOU_THRESH,
            'rpn_bg_iou_thresh': cfg.MODEL.STILL.RPN.BG_IOU_THRESH,
            'rpn_batch_size_per_image': cfg.MODEL.STILL.RPN.BATCH_SIZE_PER_IMAGE,
            'rpn_positive_fraction': cfg.MODEL.STILL.RPN.POSITIVE_FRACTION,
            'rpn_score_thresh': cfg.MODEL.STILL.RPN.SCORE_THRESH,
            'replace_head': cfg.MODEL.STILL.REPLACE_HEAD,
            'train_random_flip': cfg.TRAIN.AUGMENTATIONS.RANDOM_HORIZONTAL_FLIP,
            'roi_heads': RoiHeads(cfg)
        }

    def forward(self, images, targets=None):
        # If the dataloader has returned a list of None targets (or at least one
        # None is in the list), overwrite targets with a None for compatibility 
        if isinstance(targets, list):
            if targets[0] is None:
                targets = None
        
        if targets is not None:
            targets = [{'labels' if k=='noun_labels' else k: v for k,v in t.items()} for t in targets]
        return super().forward(images, targets)