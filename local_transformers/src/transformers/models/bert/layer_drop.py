import torch
import torch.nn as nn


class LayerDropModuleList(nn.ModuleList):
    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m
