from torchvision import transforms
import numpy as np
import torch
from detectron2.config import configurable
from random import random
import math
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform, _resize_image_and_masks, resize_boxes, resize_keypoints
from stillfast.datasets import StillFastImageTensor


class StillFastTransform(GeneralizedRCNNTransform):
    @configurable
    def __init__(self, *args, 
                train_horizontal_flip=False, 
                fast_image_mean=None,
                fast_image_std=None,
                fast_to_still_size_ratio=None,
                resize_per_batch=False,
            **kwargs):
        self.train_horizontal_flip = train_horizontal_flip
        self.fast_image_mean = fast_image_mean
        self.fast_image_std = fast_image_std
        self.fast_to_still_size_ratio = fast_to_still_size_ratio
        self.resize_per_batch = resize_per_batch
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg):
        return {
            'train_horizontal_flip': cfg.TRAIN.AUGMENTATIONS.RANDOM_HORIZONTAL_FLIP,
            'min_size': cfg.DATA.STILL.MIN_SIZE,
            'max_size': cfg.DATA.STILL.MAX_SIZE,
            'image_mean': cfg.DATA.STILL.MEAN,
            'image_std': cfg.DATA.STILL.STD,
            'fast_image_mean': cfg.DATA.FAST.MEAN,
            'fast_image_std': cfg.DATA.FAST.STD,
            'fast_to_still_size_ratio': cfg.DATA.STILL.FAST_TO_STILL_SIZE_RATIO,
            'resize_per_batch': cfg.TRAIN.GROUP_BATCH_SAMPLER
        }

    def flip_boxes(self, boxes, w):
        a = boxes[:,2].clone()
        b = boxes[:,0].clone()
        boxes[:, 0] = w - a
        boxes[:, 2] = w - b

        return boxes

    def resize(
        self,
        image,
        target,
        size
    ):
        h, w = image.shape[-2:]
        image, target = _resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

    def forward(self, data, targets=None):
        still_imgs = [d[0] for d in data]
        fast_imgs = [d[1] for d in data]
        
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy = []
            for t in targets:
                data = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy

        # Flip horizontally images and boxes
        if self.training and self.train_horizontal_flip:
            fast_imgs_out = []
            still_imgs_out = []
            targets_out = []
            
            for fast_img, still_img, target in zip(fast_imgs, still_imgs, targets):
                if random() < 0.5:
                    still_img_out = still_img.flip(2) #C x H x W
                    fast_img_out = fast_img.flip(3) #C x F x H x W
                    target_out = {
                        k: self.flip_boxes(v, still_img.shape[2]) if k=='boxes' else v for k,v in target.items()
                    }
                    fast_imgs_out.append(fast_img_out)
                    still_imgs_out.append(still_img_out)
                    targets_out.append(target_out)
                else:
                    fast_imgs_out.append(fast_img)
                    still_imgs_out.append(still_img)
                    targets_out.append(target)
        
            fast_imgs = fast_imgs_out
            still_imgs = still_imgs_out
            targets = targets_out

    
        # Generate a global size
        if self.training: 
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])

        for i in range(len(fast_imgs)):
            fast_img = fast_imgs[i]
            still_img = still_imgs[i]

            target_index = targets[i] if targets is not None else None

            if still_img.dim() != 3:
                raise ValueError(f"Still images are expected to be a list of 3d tensors of shape [C, H, W], got {still_img.shape}")
            if fast_img.dim() != 4:
                raise ValueError(f"Fast tensors are expected to be a list of 4d tensors of shape [F, C, H, W], got {fast_img.shape}")
            
            still_img = self.normalize(still_img)
            fast_img = self.normalize_fast(fast_img)

            if self.training and not self.resize_per_batch:
                size = float(self.torch_choice(self.min_size))

            still_img, target_index = self.resize(still_img, target_index, size)
            fast_img = self.resize_fast(fast_img, still_img.shape)

            fast_imgs[i] = fast_img
            still_imgs[i] = still_img
            if targets is not None and target_index is not None:
                targets[i] = target_index
        
        still_img_sizes = [img.shape[-2:] for img in still_imgs]
        fast_img_sizes = [img.shape[-2:] for img in fast_imgs]

        #if (self.training and self.apply_padding_train) or (not self.training and self.apply_padding_val):
        still_imgs = self.batch_images(still_imgs, size_divisible=self.size_divisible)
        fast_imgs = self.batch_fast_images(fast_imgs, size_divisible=self.size_divisible)
        #else:
        #    still_imgs = torch.stack(still_imgs, 0)
        #    fast_imgs = torch.stack(fast_imgs, 0)

        still_image_sizes_list = []
        for image_size in still_img_sizes:
            assert len(image_size) == 2
            still_image_sizes_list.append((image_size[0], image_size[1]))

        fast_image_sizes_list = []
        for image_size in fast_img_sizes:
            assert len(image_size) == 2
            fast_image_sizes_list.append((image_size[0], image_size[1]))

        stillfast_image_list = ImageList(
            StillFastImageTensor(still_imgs, fast_imgs),
            still_image_sizes_list
        )

        return stillfast_image_list, targets

    def normalize_fast(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None, None]) / std[:, None, None, None]

    def resize_fast(self, tensor, still_shape):
        still_size = still_shape[-2]
        target_size = int(still_size * self.fast_to_still_size_ratio)
        scale_factor = target_size / tensor.shape[-2] #C x F x H x W

        tensor = torch.nn.functional.interpolate(
            tensor,
            scale_factor = scale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=True
        )

        return tensor

    def batch_fast_images(self, images, size_divisible: int = 32):
        # FIXME: add support for ONNX export
        # if torchvision._is_tracing():
        #     # batch_images() does not export well to ONNX
        #     # call _onnx_batch_images() instead
        #     return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img[0][0].shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        
        max_size[0] = int(math.ceil(float(max_size[0]) / stride) * stride)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)

        batch_shape = [len(images), 3, images[0].shape[1]] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, :, :img.shape[1], :img.shape[2], :img.shape[3]].copy_(img)

        return batched_imgs