import torch
from copy import deepcopy as copy

class PackedTensorDictionary:
    def __init__(self, dictionaries, device='cpu'):
        self.store = dict()
        keys = list(dictionaries[0].keys())
        self._num = torch.Tensor([d[keys[0]].shape[0] for d in dictionaries]).to(device)
        for k in keys:
            self.store[k] = torch.cat([d[k] for d in dictionaries], dim=0).to(device)
        self.device = device

    def unpack(self):
        out = dict()
        for k in self.store.keys():
            out[k] = [x.cpu().numpy() for x in self.store[k].split(self.num, dim=0)]
        return out
    
    @property
    def num(self):
        return [int(x) for x in self._num]

    def __add__(self, other):
        c = copy(self)
        for k in c.store.keys():
            assert k in other.store.keys()
            c.store[k] = torch.cat([c.store[k], other.store[k]], dim=0)
        c._num = torch.cat([c._num, other._num], dim=0)
        return c

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def __repr__(self) -> str:
        s = f"num: {self.num}\n"
        for k in self.store.keys():
            s += f"{k}: {self.store[k]}\n"
        return s
   