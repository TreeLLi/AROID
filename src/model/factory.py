import os, json
import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.vit import vit_base
from src.model.swin import swin_small, swin_tiny
from src.model.wide_resnet import WideResNet
from src.model.preact_resnet import PreActResNet
from src.utils.printer import dprint

ARCHS = {'wresnet' : WideResNet,
         'paresnet': PreActResNet,
         'vit-b' : vit_base,
         'swin-s': swin_small,
         'swin-t': swin_tiny}


def fetch_model(arch, checkpoint=None, **config):
    if arch not in ARCHS:
        raise Exception("Invalid arch {}".format(arch))

    model = ARCHS[arch](**config)
    
    # extract checkpoint information
    if checkpoint is None:
        ck_lid = None
    else:
        if isinstance(checkpoint, tuple):
            ck_lid, checkpoint = checkpoint
        else:
            ck_lid = checkpoint
        ck_lid = parse_log_id_from_ck_url(ck_lid)

    hyper_params = model.hyperparams_log() if hasattr(model, 'hyperparams_log') else {}
    dprint('model',
           arch=arch,
           **hyper_params,
           checkpoint=ck_lid)
    
    if checkpoint is not None:
        # given checkpoint, resume the model from checkpoint
        if isinstance(checkpoint, str) and os.path.isfile(checkpoint):
            checkpoint = tc.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        
    return model

def parse_log_id_from_ck_url(ck_url):
    ck_url = ck_url.split('/')
    ck_url = ck_url[0] if len(ck_url) == 1 else ck_url[-1][:4]
    return ck_url
    

'''
Optimizer

'''

OPTIMS = {
    'sgd' : tc.optim.SGD,
    'adam' : tc.optim.Adam
}

def fetch_optimizer(optim, params, checkpoint=None, **args):
    # hyper-parameter report
    if checkpoint is not None:
        ck_lid, checkpoint = checkpoint
        ck_lid = parse_log_id_from_ck_url(ck_lid)
    else:
        ck_lid = None
    dprint('optimizer', optim=optim, checkpoint=ck_lid, **args)
    
    if optim in OPTIMS:
        optim = OPTIMS[optim](params, **args)
        if checkpoint is not None:
            optim.load_state_dict(checkpoint['optimizer'])
        return optim
    else:
        raise Exception("Invalid optimizer: {}".format(optim))

        
CRITERIA = {
    'crossentropy' : tc.nn.CrossEntropyLoss,
}

def fetch_criterion(criterion, device=None, **args):
    # hyper-parameter report
    dprint('criterion', criterion=criterion, **args)
    
    criterion = CRITERIA[criterion](**args)
    if device is not None:
        criterion = criterion.to(device)
    return criterion
