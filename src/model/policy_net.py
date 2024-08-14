import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

from src.model.preact_resnet import PreActResNet
from src.model.wide_resnet import WideResNet
from src.model.vit import vit_base

class PolicyNet(nn.Module):
    def __init__(self, backbone, dims, **kwargs):
        super(PolicyNet, self).__init__()

        if backbone == 'prn18':
            self.backbone = PreActResNet(depth=18, out_dim=1)
        elif 'wrn' in backbone:
            depth = int(backbone[3:])
            self.backbone = WideResNet(depth=depth, width=1, out_dim=1)
        elif backbone == 'vit-b':
            self.backbone = vit_base(input_dim=kwargs['input_dim'],
                                     out_dim=1,
                                     patch_size=kwargs['patch_size'],
                                     pretrained=True)
        else:
            raise Exception(f'invalid backbone for policy net: {backbone}')
        
        self.dims = dims
        for k, dim in dims.items():
            setattr(self, k, nn.Linear(self.backbone.num_features, dim))

    def forward(self, x):
        out = self.backbone.features(x)

        return {k : getattr(self, k)(out) for k in self.dims.keys()}

    def sampler(self, logits):
        return {k : Categorical(logits=logit) for k, logit in logits.items()}
