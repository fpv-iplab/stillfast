import pytorch_lightning as pl
import torch
from torchvision.ops import box_iou
from .base_task import BaseTask
from stillfast.evaluation.sta_metrics import OverallMeanAveragePrecision
import itertools
import json
from stillfast.tasks.utils import PackedTensorDictionary
from stillfast.utils.distributed import list_gather

class STATask(BaseTask):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self.cfg = cfg

        self.learning_rate = self.cfg.SOLVER.BASE_LR
        self.checkpoint_metric = 'map_box_noun_verb_ttc'

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss_dict = self.model(batch)
        loss = sum(loss for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v.item())
        self.log('train/loss_overall', loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        preds = self.model(batch)

        return {
            'uids': batch['uids'],
            'preds': PackedTensorDictionary(preds)
        }

    def test_epoch_end(self, outs):
        outs = list_gather(outs)
        uids = list(itertools.chain(*[o['uids'] for o in outs]))
        preds = sum([o['preds'] for o in outs]).unpack()
        
        pred_boxes = preds['boxes']
        pred_nouns = preds['nouns']
        pred_verbs = preds['verbs']
        pred_ttcs = preds['ttcs']
        pred_scores = preds['scores']
        
        pred_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
                "verbs": verbs,
                "ttcs": ttcs,
                "scores": scores
            } for uid, boxes, nouns, verbs, ttcs, scores in zip(uids, pred_boxes, pred_nouns, pred_verbs, pred_ttcs, pred_scores)
        }

        # FIXME Check if verbs and nouns are in the correct range
        # The dataloader returns nouns shifted by 1. When ROIHeadsV2 is used, 
        # also verbs are shifted by 1. Not sure this complies with official annotations
        # on EvalAI
        if self.cfg.TEST.OUTPUT_JSON:
            output_dict = {
                'version': '1.0',
                'challenge': 'ego4d_short_term_object_interaction_anticipation',
                'results' : {}
            }
            if 'v1' not in self.cfg.MODEL.STILLFAST.ROI_HEADS.VERSION:
                verb_offset = 1
            else:
                verb_offset = 0
            for uid, pred in pred_detections.items():
                output_dict['results'][uid] = []
                for box, noun, verb, ttc, score in zip(pred['boxes'], pred['nouns'], pred['verbs'], pred['ttcs'], pred['scores']):
                    output_dict['results'][uid].append({
                        'box': [float(b) for b in box],
                        'score': float(score),
                        'noun_category_id': int(noun)-1,
                        'verb_category_id': int(verb)-verb_offset,
                        'time_to_contact': float(ttc)
                    })
            with open(self.cfg.TEST.OUTPUT_JSON, 'w') as f:
                json.dump(output_dict, f)

        print(f"Test done on {len(pred_detections)} predictions, {sum([len(b) for b in pred_boxes])} boxes in total")

    def validation_step(self, batch, batch_idx):

        preds = self.model(batch)
        

        return {
            'uids': batch['uids'],
            'preds': PackedTensorDictionary(preds),
            'targets': PackedTensorDictionary(batch['targets'])
        }

    def validation_epoch_end(self, outs):
        outs = list_gather(outs)
        uids = list(itertools.chain(*[o['uids'] for o in outs]))
        preds = sum([o['preds'] for o in outs]).unpack()
        targets = sum([o['targets'] for o in outs]).unpack()
        
        pred_boxes = preds['boxes']
        pred_nouns = preds['nouns']
        pred_verbs = preds['verbs']
        pred_ttcs = preds['ttcs']
        pred_scores = preds['scores']
        gt_boxes = targets['boxes']
        gt_nouns = targets['noun_labels']
        gt_verbs = targets['verb_labels']
        gt_ttcs = targets['ttc_targets']
        
        pred_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
                "verbs": verbs,
                "ttcs": ttcs,
                "scores": scores
            } for uid, boxes, nouns, verbs, ttcs, scores in zip(uids, pred_boxes, pred_nouns, pred_verbs, pred_ttcs, pred_scores)
        }

        gt_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
                "verbs": verbs,
                "ttcs": ttcs
            } for uid, boxes, nouns, verbs, ttcs in zip(uids, gt_boxes, gt_nouns, gt_verbs, gt_ttcs)
        }

        map = OverallMeanAveragePrecision(top_k=5)
        for uid in gt_detections.keys():
            map.add(pred_detections[uid], gt_detections[uid])

        vals = map.evaluate()
        names = map.get_short_names()

        # FIXME Check if verbs and nouns are in the correct range
        # The dataloader returns nouns shifted by 1. When ROIHeadsV2 is used, 
        # also verbs are shifted by 1. Not sure this complies with official annotations
        # on EvalAI
        if self.cfg.VAL.OUTPUT_JSON:
            output_dict = {
                'version': '1.0',
                'challenge': 'ego4d_short_term_object_interaction_anticipation',
                'results' : {}
            }
            if 'v1' not in self.cfg.MODEL.STILLFAST.ROI_HEADS.VERSION:
                verb_offset = 1
            else:
                verb_offset = 0
            for uid, pred in pred_detections.items():
                output_dict['results'][uid] = []
                for box, noun, verb, ttc, score in zip(pred['boxes'], pred['nouns'], pred['verbs'], pred['ttcs'], pred['scores']):
                    output_dict['results'][uid].append({
                        'box': [float(b) for b in box],
                        'score': float(score),
                        'noun_category_id': int(noun)-1,
                        'verb_category_id': int(verb)-verb_offset,
                        'time_to_contact': float(ttc)
                    })
            with open(self.cfg.VAL.OUTPUT_JSON, 'w') as f:
                json.dump(output_dict, f)

        for name, val in zip(names, vals):
            self.log(f"val/{name}", val)
            if name==self.checkpoint_metric:
                self.log(f"{name}", val)

        print(f"Validation done on {len(pred_detections)} predictions, {sum([len(b) for b in pred_boxes])} boxes in total")