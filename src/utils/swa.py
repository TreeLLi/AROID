'''
Code from Robust overfitting can be alliviated 

'''

import torch as tc

from src.data.policy import Policy

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, tc.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, tc.nn.modules.batchnorm._BatchNorm):
        module.running_mean = tc.zeros_like(module.running_mean)
        module.running_var = tc.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, tc.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, tc.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loaders, models, args):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(models.swa):
        return
    models.swa.train()
    momenta = {}
    models.swa.apply(reset_bn)
    models.swa.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    policy = Policy(args.out_dim)
    
    extra_loader = iter(loaders.extra) if 'extra' in loaders else None
    
    for imgs, tgts in loaders.train:
        if extra_loader is not None:
            eimgs, etgts = next(extra_loader)
            imgs = tc.cat((imgs, eimgs))
            tgts = tc.cat((tgts, etgts))

        imgs = imgs.to(args.device, non_blocking=True)
        tgts = tgts.to(args.device, non_blocking=True)
        
        b = imgs.data.size(0)
        
        with tc.inference_mode():
            policy.dists = models.policy.sampler(models.policy(imgs))

        imgs, _, _ = policy(imgs, tgts)
    
        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        models.swa(imgs)
        n += b

    models.swa.apply(lambda module: _set_momenta(module, momenta))
