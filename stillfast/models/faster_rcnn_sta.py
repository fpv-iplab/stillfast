"""Implements a full baseline for Short-Term object interaction Anticipation (STA)
based on a (2D) Faster-RCNN model."""

from .faster_rcnn import FasterRCNN, RoiHeads
from torch import nn
from detectron2.config import configurable
from torch.nn import functional as F
from stillfast.models.losses import get_loss_func
from .build import MODEL_REGISTRY
import torch
from torchvision.ops import boxes as box_ops
import numpy as np
from typing import List, Dict
from torchvision.models.detection.roi_heads import fastrcnn_loss


class RoiHeadsSTA(RoiHeads):
    @configurable
    def __init__(self, verb_predictor, ttc_predictor, loss_weights, loss_verb, loss_ttc, **kwargs):
        super().__init__(**kwargs)
        self.verb_predictor = verb_predictor
        self.ttc_predictor = ttc_predictor
        self.loss_weights = loss_weights
        self.loss_verb = get_loss_func(loss_verb)()
        self.loss_ttc = get_loss_func(loss_ttc)()
        self.box_head.register_forward_hook(self._box_head_forward_hook)

    def select_training_samples(self, proposals, targets):
        proposals, matched_idxs, labels, regression_targets = super().select_training_samples(proposals, targets)
        self.matched_idxs = matched_idxs
        self.noun_labels = labels
        return proposals, matched_idxs, labels, regression_targets

    def _box_head_forward_hook(self, module, input, output):
        self.box_features = output

    def postprocess_detections(
        self,
        class_logits,  
        box_regression,
        proposals,  
        image_shapes, 
    ):
        """Override origial method just to save inputs and return empty results.
        The actual postprocessing will be performed in a dedicated method which
        can access also other inputs."""

        self.class_logits = class_logits
        self.box_regression = box_regression
        self.proposals = proposals
        self.image_shapes = image_shapes

        return [], [], []

    def postprocess_detections_sta(
        self,
        noun_logits,
        verb_logits,
        ttc_predictions,
        box_regression,
        proposals,
        image_shapes
    ):
        device = noun_logits.device
        num_classes = noun_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(noun_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        verb_predictions = verb_logits.argmax(-1)
        verb_predictions_list = verb_predictions.split(boxes_per_image, 0)
        ttc_predictions_list = ttc_predictions.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_nouns = []
        all_verbs = []
        all_ttcs = []
        for boxes, scores, image_shape, verbs, ttcs in zip(pred_boxes_list, pred_scores_list, image_shapes, verb_predictions_list, ttc_predictions_list):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            keep_idx = torch.arange(boxes.shape[0], device=device)
            keep_idx = keep_idx.view(-1,1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            keep_idx = keep_idx[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            keep_idx = keep_idx.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, keep_idx = boxes[inds], scores[inds], labels[inds], keep_idx[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, keep_idx = boxes[keep], scores[keep], labels[keep], keep_idx[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, keep_idx = boxes[keep], scores[keep], labels[keep], keep_idx[keep]

            verbs, ttcs = verbs[keep_idx], ttcs[keep_idx]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_nouns.append(labels)
            all_verbs.append(verbs)
            all_ttcs.append(ttcs)

        result: List[Dict[str, torch.Tensor]] = []
        num_images = len(all_boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": all_boxes[i],
                    "nouns": all_nouns[i],
                    "verbs": all_verbs[i],
                    "ttcs": all_ttcs[i],
                    "scores": all_scores[i],
                }
            )    
        
        return result

    def forward(self, features, proposals, image_shapes, targets=None):
        result, losses = super().forward(features, proposals, image_shapes, targets)
        verb_logits = self.verb_predictor(self.box_features)
        ttc_predictions = self.ttc_predictor(self.box_features)

        if self.training:
            # Select targets
            num_images = len(proposals)
            verb_labels = [None]*num_images
            ttc_targets = [None]*num_images
            valid_targets = [None]*num_images
            for img_id in range(num_images):
                idx = self.matched_idxs[img_id]
                vl = targets[img_id]['verb_labels']
                tt = targets[img_id]['ttc_targets']
                verb_labels[img_id] = vl[idx]
                ttc_targets[img_id] = tt[idx]
                valid_targets[img_id] = self.noun_labels[img_id] != 0
            verb_labels = torch.cat(verb_labels, dim=0)
            ttc_targets = torch.cat(ttc_targets, dim=0).unsqueeze(-1)
            valid_targets = torch.cat(valid_targets, dim=0)
            
            # rename loss_classifier to loss_noun
            losses['loss_noun'] = self.loss_weights.NOUN*losses.pop('loss_classifier')
            losses.update({
                "loss_verb": self.loss_weights.VERB*self.loss_verb(verb_logits[valid_targets], verb_labels[valid_targets]),
                "loss_ttc": self.loss_weights.TTC*self.loss_ttc(ttc_predictions[valid_targets], ttc_targets[valid_targets])
                }
            )
        else:
            result = self.postprocess_detections_sta(
                self.class_logits,
                verb_logits,
                ttc_predictions,
                self.box_regression,
                self.proposals,
                self.image_shapes
            )

        return result, losses

    @classmethod
    def from_config(cls, cfg):
        representation_size = cfg.MODEL.STILL.BOX.PREDICTOR_REPRESENTATION_SIZE
        num_verbs = cfg.MODEL.VERB_CLASSES

        options = super().from_config(cfg)
        verb_predictor = nn.Linear(representation_size, num_verbs)

        if cfg.MODEL.TTC_PREDICTOR == 'regressor':
            ttc_predictor = nn.Sequential(nn.Linear(representation_size, 1), nn.Softplus())
        else:
            raise NotImplementedError

        options.update({
            'verb_predictor': verb_predictor,
            'ttc_predictor': ttc_predictor,
            'loss_weights': cfg.MODEL.LOSS.WEIGHTS,
            'loss_verb': cfg.MODEL.LOSS.VERB,
            'loss_ttc': cfg.MODEL.LOSS.TTC,
        })
            
        return options



class RoiHeadsSTAv2(RoiHeadsSTA):
    """Version 2 of the head. Implements the final head documented in the associated paper."""

    @configurable
    def __init__(self, *args, fusion='sum', verb_topk=1, **kwargs):
        super().__init__(*args, **kwargs)
        print("ROIHEADSSTAv2")
        self.fusion = fusion
        self.verb_topk = verb_topk
        if fusion == 'sum':
            self.mapper = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024)
                )
        elif 'concat' in fusion:
            self.mapper = nn.Sequential(
                nn.Linear(256 + 1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024)
                )

        if fusion=='concat_residual':
            self.residual = True
            self.fusion = 'concat'
        else:
            self.residual = False

    @classmethod
    def from_config(cls, cfg):
        representation_size = cfg.MODEL.STILL.BOX.PREDICTOR_REPRESENTATION_SIZE
        num_verbs = cfg.MODEL.VERB_CLASSES

        options = super().from_config(cfg)
        # Add background class to verb predictor
        verb_predictor = nn.Linear(representation_size, num_verbs+1)

        options.update({
            'verb_predictor': verb_predictor,
            'fusion': cfg.MODEL.STILLFAST.ROI_HEADS.V2_OPTIONS.FUSION,
            'verb_topk': cfg.MODEL.STILLFAST.ROI_HEADS.V2_OPTIONS.VERB_TOPK
        })
            
        return options

    def _check_targets(self, targets):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, "target keypoints must of float type"

    def _select_training_samples(self, proposals, targets):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        return proposals, matched_idxs, labels, regression_targets

    def _compute_box_features(self, features, proposals, image_shapes):
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        return box_features

    def forward(
        self,
        features,  
        proposals, 
        image_shapes,
        targets=None, 
    ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        self._check_targets(targets)
        proposals, matched_idxs, noun_labels, regression_targets = self._select_training_samples(proposals, targets)
        box_features = self._compute_box_features(features, proposals, image_shapes)

        global_features = features['pool']
        global_features = global_features.view(global_features.shape[0], global_features.shape[1],-1).mean(-1)
        
        BS = global_features.shape[0]

        if self.fusion == 'sum':
            gl_box_features = box_features.reshape(BS, -1, box_features.shape[-1])
            gl_box_features = gl_box_features + self.mapper(global_features).unsqueeze(1).expand(BS, gl_box_features.shape[1], gl_box_features.shape[-1])
            gl_box_features = gl_box_features.reshape(-1, box_features.shape[-1])
        elif self.fusion == 'concat':
            gl_box_features = box_features.reshape(BS, -1, box_features.shape[-1]) # [BS*NUM_PROPOSALS, 1024] -> [BS, NUM_PROPOSALS, 1024]
            global_features = global_features.unsqueeze(1).expand(BS, gl_box_features.shape[1], global_features.shape[-1]) # [BS, 256] -> [BS, NUM_PROPOSALS, 256]
            gl_box_features = torch.cat([gl_box_features, global_features], dim=-1) # [BS, NUM_PROPOSALS, 1024+256]
            gl_box_features = self.mapper(gl_box_features).reshape(-1, box_features.shape[-1]) # [BS*NUM_PROPOSALS, 1024+256] -> [BS*NUM_PROPOSALS, 1024]
            if self.residual:
                gl_box_features = box_features + gl_box_features
        else:
            raise ValueError('Unknown fusion method: {}'.format(self.fusion))
        
        class_logits, box_regression = self.box_predictor(gl_box_features)
        verb_logits = self.verb_predictor(gl_box_features)
        ttc_predictions = self.ttc_predictor(gl_box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}

        if self.training:
            assert noun_labels is not None and regression_targets is not None
            # Select targets
            num_images = len(proposals)
            verb_labels = [None]*num_images # fill targets with None
            ttc_targets = [None]*num_images # fill targets with None
            valid_targets = [None]*num_images # fill valid targets with None
            for img_id in range(num_images): # for each image
                idx = matched_idxs[img_id] # get the index of the matched proposals
                vl = targets[img_id]['verb_labels'] # get the verb labels of the current image
                tt = targets[img_id]['ttc_targets'] # get the ttc targets of the current image
                verb_labels[img_id] = vl[idx] # attach the verb label of the matched gt
                ttc_targets[img_id] = tt[idx] # attach the ttc target of the matched gt
                valid_targets[img_id] = noun_labels[img_id] != 0 # valid targets are such that the noun label is not 0
            verb_labels = torch.cat(verb_labels, dim=0)
            ttc_targets = torch.cat(ttc_targets, dim=0).unsqueeze(-1)
            valid_targets = torch.cat(valid_targets, dim=0)

            # we now want to set all verbs for invalid matches to the background class zero
            verb_labels[~valid_targets] = 0
            
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, noun_labels, regression_targets)
            losses = {
                "loss_noun": self.loss_weights.NOUN*loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_verb": self.loss_weights.VERB*self.loss_verb(verb_logits, verb_labels),
                "loss_ttc": self.loss_weights.TTC*self.loss_ttc(ttc_predictions[valid_targets], ttc_targets[valid_targets])
                }
        else:
            result = self.postprocess_detections_sta(
                class_logits,
                verb_logits,
                ttc_predictions,
                box_regression,
                proposals,
                image_shapes
            )

        return result, losses

    def postprocess_detections_sta(
        # Verb-noun product
        self,
        noun_logits,
        verb_logits,
        ttc_predictions,
        box_regression,
        proposals,
        image_shapes
    ):
        device = noun_logits.device
        noun_classes = noun_logits.shape[-1] # number of nouns
        verb_classes = verb_logits.shape[-1] # number of verbs

        # number of proposals per image
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals) # box predictions

        # apply softmax to get noun and verb scores
        noun_scores = F.softmax(noun_logits, -1)
        verb_scores = F.softmax(verb_logits, -1)

        # split boxes, ttc predictions, verb and noun scores in a list of per-image tensors
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        noun_scores_list = noun_scores.split(boxes_per_image, 0)
        verb_scores_list = verb_scores.split(boxes_per_image, 0)
        ttc_predictions_list = ttc_predictions.split(boxes_per_image, 0)

        # accumulate boxes, scores, verb, noun and ttcs predictions
        all_boxes = []
        all_scores = []
        all_nouns = []
        all_verbs = []
        all_ttcs = []
        # for each image, iterate over boxes noun_scores, image_shape, verb_scores, ttc predictions
        for boxes, noun_scores, image_shape, verb_scores, ttcs in zip(pred_boxes_list, noun_scores_list, image_shapes, verb_scores_list, ttc_predictions_list):
            # clip boxes which may be outside image
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create noun labels for each prediction
            noun_labels = torch.arange(noun_classes, device=device)
            noun_labels = noun_labels.view(1, -1).expand_as(noun_scores) # N x num_nouns

            # create noun labels for each prediction
            verb_labels = torch.arange(verb_classes, device=device)
            verb_labels = verb_labels.view(1, -1).expand_as(verb_scores) # N x num_verbs

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            noun_scores = noun_scores[:, 1:]
            noun_labels = noun_labels[:, 1:]

            verb_scores = verb_scores[:, 1:]
            verb_labels = verb_labels[:, 1:]


            ### TRUNCATE verbs to topk
            K = self.verb_topk
            # # sort each verb prediction by scores and keep only the top K
            verb_scores, verb_idx = verb_scores.sort(-1, descending=True)
            verb_labels = torch.stack([verb_labels[i,verb_idx[i]] for i in range(verb_idx.shape[0])])
            verb_scores = verb_scores[:,:K]
            verb_labels = verb_labels[:,:K]
            
            # now we have the top verb and noun scores and labels for each box
            # Let's compute a N x K x K matrix of K x K noun-verb scores
            vn_scores= noun_scores.unsqueeze(-1) * verb_scores.unsqueeze(-2)
            noun_labels = noun_labels.unsqueeze(2).expand_as(vn_scores) # Expand to be N x K x 1 -> N x K x K
            verb_labels = verb_labels.unsqueeze(1).expand_as(vn_scores) # Expand to be N x 1 x K -> N x K x K
            _, A, B = vn_scores.shape
            boxes = boxes.unsqueeze(2).expand(boxes.shape[0], A, B, boxes.shape[-1]) # Expand to be N x K x 1 x 4 -> N x K x K x 4

            # Let's create a keep_idx array based on this grid of predictions
            # This will be N x K x K and indicating the rows of boxes and ttcs to index
            keep_idx = torch.arange(boxes.shape[0], device=device) # N
            keep_idx = keep_idx.view(-1,1,1).expand_as(vn_scores) # N x 1 x 1 -> N x K x K

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4) # N x 4
            vn_scores = vn_scores.reshape(-1) # N*K*K
            keep_idx = keep_idx.reshape(-1) # N*K*K
            verb_labels = verb_labels.reshape(-1)
            noun_labels = noun_labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(vn_scores > self.score_thresh)[0]
            boxes, vn_scores, noun_labels, verb_labels, keep_idx = boxes[inds], vn_scores[inds], noun_labels[inds], verb_labels[inds], keep_idx[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, vn_scores, noun_labels, verb_labels, keep_idx = boxes[keep], vn_scores[keep], noun_labels[keep], verb_labels[keep], keep_idx[keep]

            vn_labels = noun_labels + verb_labels*(noun_classes+1)
            
            # non-maximum suppression, independently done per verb-noun class
            keep = box_ops.batched_nms(boxes, vn_scores, vn_labels, self.nms_thresh)
            
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, vn_scores, noun_labels, verb_labels, keep_idx = boxes[keep], vn_scores[keep], noun_labels[keep], verb_labels[keep], keep_idx[keep]

            ttcs = ttcs[keep_idx]

            all_boxes.append(boxes)
            all_scores.append(vn_scores)
            all_nouns.append(noun_labels)
            all_verbs.append(verb_labels)
            all_ttcs.append(ttcs)

        result: List[Dict[str, torch.Tensor]] = []
        num_images = len(all_boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": all_boxes[i],
                    "nouns": all_nouns[i],
                    "verbs": all_verbs[i],
                    "ttcs": all_ttcs[i],
                    "scores": all_scores[i],
                }
            )    
        
        return result
    
