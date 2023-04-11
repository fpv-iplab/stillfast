import numpy as np
import torch

def get_random_size(cfg, train):
    if train:
            still_side = int(np.random.randint(cfg.DATA.MIN_SIZE_TRAIN_STILL, cfg.DATA.MAX_SIZE_TRAIN_STILL, 1)[0])
    else:
        still_side = cfg.DATA.SIZE_TEST_STILL

    return still_side

def flip_boxes(gt_boxes, orig_sw):
    a = gt_boxes[:,2].clone()
    b = gt_boxes[:,0].clone()
    gt_boxes[:, 0] = orig_sw - a
    gt_boxes[:, 2] = orig_sw - b
    
    return gt_boxes

def get_crop_bounds(sh, sw, still_crop):
    if sw-still_crop==0:
        still_minx=0
    else:
        still_minx = np.random.randint(0, sw-still_crop, 1)[0]

    if sh-still_crop==0:
        still_miny=0
    else:
        still_miny = np.random.randint(0, sh-still_crop, 1)[0]

    return still_minx, still_miny

def crop_boxes(gt_boxes, still_minx, still_miny):
    b = gt_boxes.clone()
    gt_boxes[:, 0] = b[:,0] - still_minx
    gt_boxes[:, 1] = b[:,1] - still_miny
    gt_boxes[:, 2] = b[:,2] - still_minx
    gt_boxes[:, 3] = b[:,3] - still_miny
    return gt_boxes

def clip_boxes(gt_boxes, still_img):
    gt_boxes[:, 0] = torch.clip(gt_boxes[:, 0], 0, still_img.shape[2])
    gt_boxes[:, 1] = torch.clip(gt_boxes[:, 1], 0, still_img.shape[1])
    gt_boxes[:, 2] = torch.clip(gt_boxes[:, 2], 0, still_img.shape[2])
    gt_boxes[:, 3] = torch.clip(gt_boxes[:, 3], 0, still_img.shape[1])

    areas = (gt_boxes[:,2] - gt_boxes[:,0])*(gt_boxes[:,3] - gt_boxes[:,1])
    gt_boxes = gt_boxes[areas>0] # remove empty boxes
    
    return gt_boxes

def crop_img(img, minx, miny, crop):
    return img[:, miny:miny+crop, minx:minx+crop]