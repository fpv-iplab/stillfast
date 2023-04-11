import torch
import json
import os.path
import numpy as np
from PIL import Image
import io
from typing import List

from .build import DATASET_REGISTRY
from stillfast.datasets.sta_hlmdb import Ego4DHLMDB
from stillfast.datasets.ego4d_sta_still import Ego4dShortTermAnticipationStill
from stillfast.datasets import StillFastImageTensor

class Ego4DHLMDB_STA_Still_Video(Ego4DHLMDB):
    def get(self, video_id: str, frame: int) -> np.ndarray:
        with self._get_parent(video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())

                return Image.open(io.BytesIO(data))
    
    def get_batch(self, video_id: str, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(video_id) as env:
            with env.begin() as txn:
                for frame in frames:
                    #print("video_id", video_id, "Frame", frame)
                    data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())
                    out.append(Image.open(io.BytesIO(data)))
            return out

# TODO: refactor as reconfigurable
@DATASET_REGISTRY.register()
class Ego4dShortTermAnticipationStillVideo(Ego4dShortTermAnticipationStill):
    """
    Ego4d Short Term Anticipation StillVideo Dataset
    """

    def __init__(self, cfg, split):
        super(Ego4dShortTermAnticipationStillVideo, self).__init__(cfg, split)
        self._fast_hlmdb = Ego4DHLMDB_STA_Still_Video(self.cfg.EGO4D_STA.FAST_LMDB_PATH, readonly=True, lock=False)

    def _load_frames_lmdb(self, video_id, frames):
        """ Load images from lmdb. """
        imgs = self._fast_hlmdb.get_batch(video_id, frames)
        return imgs

    def _sample_frames(self, frame):
        """ Sample frames from a video. """
        frames = (
                frame
                - np.arange(
            self.cfg.DATA.FAST.NUM_FRAMES * self.cfg.DATA.FAST.SAMPLING_RATE,
            step=self.cfg.DATA.FAST.SAMPLING_RATE,
            )[::-1]
        )
        frames[frames < 0] = 0

        frames = frames.astype(int)

        return frames

    def _load_still_fast_frames(self, video_id, frame_number):
        """ Load frames from video_id and frame_number """
        frames_list = self._sample_frames(frame_number)

        fast_imgs = self._load_frames_lmdb(
                video_id, frames_list
            )

        still_img = self._load_still_frame(video_id, frame_number)

        return still_img, fast_imgs
    
    def __getitem__(self, idx):
        """ Get the idx-th sample. """
        uid, video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets = self._load_annotations(idx)

        still_img, fast_imgs = self._load_still_fast_frames(video_id, frame_number)

        still_img = self.convert_tensor(still_img)
        fast_imgs = torch.stack([self.convert_tensor(img) for img in fast_imgs], dim=1)

        # FIXME: this is a hack to make the dataset compatible with the original Ego4d dataset
        # This could create problems when producing results on the test set and sending them to the
        # evaluation server.
        if 'v1' not in self.cfg.MODEL.STILLFAST.ROI_HEADS.VERSION:
            verb_offset = 1
        else:
            verb_offset = 0
            
        targets = {
            'boxes': torch.from_numpy(gt_boxes),
            'noun_labels': torch.Tensor(gt_noun_labels).long()+1,
            'verb_labels': torch.Tensor(gt_verb_labels).long()+verb_offset,
            'ttc_targets': torch.Tensor(gt_ttc_targets)
        } if gt_boxes is not None else None

        return {'still_img': still_img, 'fast_imgs': fast_imgs, 'targets': targets, 'uids': uid}