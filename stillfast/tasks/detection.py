import pytorch_lightning as pl
import torch
from torchvision.ops import box_iou
from .base_task import BaseTask
from stillfast.evaluation.sta_metrics import ObjectOnlyMeanAveragePrecision
import itertools
import json

class SimpleDetectionTask(BaseTask):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self.cfg = cfg

        self.learning_rate = self.cfg.SOLVER.BASE_LR
        self.checkpoint_metric = 'val/map_box_noun'

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['images'] #batch['images']
        targets = batch['targets']
        #targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v.item())
        self.log('train/loss_overall', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['images']
        targets = batch['targets']
        uids = batch['uids']
        #targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)

        pred_boxes = [o['boxes'].cpu().numpy() for o in outs]
        pred_nouns = [o['labels'].cpu().numpy() for o in outs]
        pred_scores = [o['scores'].cpu().numpy() for o in outs]

        gt_boxes = [t['boxes'].cpu().numpy() for t in targets]
        gt_nouns = [t['noun_labels'].cpu().numpy() for t in targets]

        return {
            "uids" : list(uids),
            "pred_boxes" : pred_boxes,
            "pred_nouns" : pred_nouns,
            "pred_scores" : pred_scores,
            "gt_boxes" : gt_boxes,
            "gt_nouns" : gt_nouns,
        }

    def validation_epoch_end(self, outs):
        uids = list(itertools.chain(*[o['uids'] for o in outs]))
        pred_boxes = list(itertools.chain.from_iterable([o['pred_boxes'] for o in outs]))
        pred_nouns = list(itertools.chain.from_iterable([o['pred_nouns'] for o in outs]))
        pred_scores = list(itertools.chain.from_iterable([o['pred_scores'] for o in outs]))
        gt_boxes = list(itertools.chain.from_iterable([o['gt_boxes'] for o in outs]))
        gt_nouns = list(itertools.chain.from_iterable([o['gt_nouns'] for o in outs]))

        pred_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
                "scores": scores
            } for uid, boxes, nouns, scores in zip(uids, pred_boxes, pred_nouns, pred_scores)
        }

        gt_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
            } for uid, boxes, nouns in zip(uids, gt_boxes, gt_nouns)
        }

        map = ObjectOnlyMeanAveragePrecision(top_k=5)
        for uid in gt_detections.keys():
            map.add(pred_detections[uid], gt_detections[uid])

        vals = map.evaluate()
        names = map.get_short_names()

        if self.cfg.VAL.OUTPUT_JSON:
            output_dict = {
                'version': '1.0',
                'challenge': 'ego4d_short_term_object_interaction_anticipation',
                'results' : {}
            }
            for uid, pred in pred_detections.items():
                output_dict['results'][uid] = []
                for box, noun, score in zip(pred['boxes'], pred['nouns'], pred['scores']):
                    output_dict['results'][uid].append({
                        'box': [float(b) for b in box],
                        'score': float(score),
                        'noun_category_id': int(noun)
                    })
            with open(self.cfg.VAL.OUTPUT_JSON, 'w') as f:
                json.dump(output_dict, f)

        for name, val in zip(names, vals):
            self.log(f"val/{name}", val)