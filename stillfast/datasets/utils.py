from torch import nn
import numpy as np
import pandas as pd
from copy import copy

class StillFastImageTensor(nn.Module):
    def __init__(self, still_tensor, fast_tensor):
        super().__init__()
        self.still_tensor = still_tensor
        self.fast_tensor = fast_tensor
    
    @property
    def shape(self):
        return self.still_tensor.shape
    
    @property
    def device(self):
        return self.still_tensor.device
    
    def __repr__(self):
        return f"(still_tensor = {str(self.still_tensor)}\nfast_tensor = {str(self.fast_tensor)})"
    
    def __len__(self):
        return 2
    
    def __getitem__(self, idx):
        if idx==0:
            return self.still_tensor
        elif idx==1:
            return self.fast_tensor
        else:
            raise IndexError("index out of range")
class ProbabilityEstimator:
    def __init__(self, x, y=None):
        x = np.array(x)
        self.x = x
        self.y = None
        if y is None:
            self.d1 = True
            if x.dtype==np.int64:
                self.hist = np.bincount(x)
                self.hist = self.hist/self.hist.sum()
                self.edges=None
            else:
                self.hist, self.edges = np.histogram(x, bins=16, density=True)
                self.hist = self.hist* (self.edges[1:]-self.edges[:-1])
        else:
            self.d1 = False
            y = np.array(y)
            self.H, self.xedges, self.yedges = np.histogram2d(x,y, density=True, bins=16)
            xsize = self.xedges[1:]-self.xedges[:-1]
            ysize = self.yedges[1:]-self.yedges[:-1]
            self.H = self.H*xsize.reshape(-1,1).dot(ysize.reshape(1,-1))
            self.y = y

    def __call__(self, x=None, y=None):
        if x is None and y is None:
            return self(self.x, self.y)
        x = np.array(x)
        if self.d1:
            assert y == None
            if self.edges is not None:
                idx = np.digitize(x, self.edges, right=True)-1
                return self.hist[idx]
            else:
                return self.hist[x]
        else:
            assert y is not None
            y = np.array(y)
            i = np.digitize(x, self.xedges, right=True)-1
            j = np.digitize(y, self.yedges, right=True)-1
            return self.H[i,j]

def get_annotations_weights(annotations):
    boxes = []
    nouns = []
    verbs = []
    ttcs = []
    uids = []
    for ann in annotations['annotations']:
        info = annotations['videos'][ann['video_id']]
        for obj in ann['objects']:
            box = copy(obj['box'])
            box[0]/=info['frame_width']
            box[1]/=info['frame_height']
            box[2]/=info['frame_width']
            box[3]/=info['frame_height']
            boxes.append(box)
            nouns.append(obj['noun_category_id'])
            verbs.append(obj['verb_category_id'])
            ttcs.append(obj['time_to_contact'])
            uids.append(ann['uid'])
            
    boxes = np.array(boxes)
    verbs = np.array(verbs)
    ttcs = np.array(ttcs)
    scales = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    center_x = boxes[:,0] + (boxes[:,2]-boxes[:,0])/2
    center_y = boxes[:,1] + (boxes[:,3]-boxes[:,1])/2

    box_probs = pd.DataFrame({
        'uid': uids,
        'prob_scale': ProbabilityEstimator(scales)(),
        'prob_noun': ProbabilityEstimator(nouns)(),
        'prob_verb': ProbabilityEstimator(verbs)(),
        'prob_ttc': ProbabilityEstimator(ttcs)(),
        'prob_position': ProbabilityEstimator(center_x, center_y)()
    })
    vc=box_probs.groupby('uid').count()['prob_scale'].value_counts()
    box_probs=box_probs.merge(box_probs.groupby('uid').count()['prob_scale'].replace((vc/vc.sum()).to_dict()).rename('prob_box'), on='uid')
    box_probs['overall_prob']=(box_probs['prob_scale']+box_probs['prob_noun']+box_probs['prob_verb']+box_probs['prob_ttc']+box_probs['prob_position']+box_probs['prob_box'])/6
    weight_dict = box_probs.set_index('uid')['overall_prob'].to_dict()

    return 1-np.array([weight_dict[x['uid']] for x in annotations['annotations']])
