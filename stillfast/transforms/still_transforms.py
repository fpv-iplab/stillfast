from re import I
from turtle import forward
from torchvision import transforms
import numpy as np
import torch
from .utils import get_random_size, flip_boxes, get_crop_bounds, clip_boxes, crop_img, crop_boxes
from torchvision.models.detection.image_list import ImageList
import math
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from random import random

from torch.nn.functional import interpolate

class GeneralizedRCNNTransformWithHorizontalFlip(GeneralizedRCNNTransform):
    def __init__(self, *args, train_horizontal_flip=False, **kwargs):
        self.train_horizontal_flip = train_horizontal_flip
        super().__init__(*args, **kwargs)
    
    def flip_boxes(self, boxes, w):
        a = boxes[:,2].clone()
        b = boxes[:,0].clone()
        boxes[:, 0] = w - a
        boxes[:, 2] = w - b
        
        return boxes

    def forward(self, images, targets=None):
        # FIXME: this should be done independently 
        # for each image, not just the batch
        if self.training and self.train_horizontal_flip and random() < 0.5:
            images = [im.flip(2) for im in images]
            targets = [{
                k: self.flip_boxes(v,im.shape[2]) if k=='boxes' else v for k,v in t.items()
            } for t,im in zip(targets, images)]
            
        return super().forward(images, targets)