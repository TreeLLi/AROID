import math
import numpy as np

import torch as tc
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode as Interpolation

from torch.nn.functional import softmax, log_softmax, one_hot

from collections import OrderedDict
from addict import Dict

COLSHA_SPACE_M = OrderedDict({
    'identity' : [],
    'autocontrast' : [],
    'equalize' : [],
    'color' : [2, 0.7, 0.3, 0.1],
    'sharpness' : [2, 0.7, 0.3, 0.01],
    'contrast' : [2, 0.92, 0.82, 0.73, 0.67, 0.62, 0.56, 0.5],
    'brightness' : [2, 0.92, 0.82, 0.75, 0.7, 0.65, 0.6, 0.56],
    'shear' : [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.7],
    'rotate' : [0, 3, 7, 12, 15, 19, 23, 31]
})

class Policy(tc.nn.Module):
    dims = Dict(OrderedDict({
        'flip' : 2,
        'crop' : 16,
        'colsha' : None,
        'erase' : 11
    }))
    
    @classmethod
    def init_colsha_space_a(cls, scheme, gran):
        space = []
        scheme = SPACE[scheme.upper()]
        mags = []
        for op, ran in scheme.items():
            pre_space_len = len(space)
            
            if ran is None:
                space.append((op, None))
            else:
                lower, upper, discrete = ran
                step = float(upper - lower) / gran

                last_mag = None
                
                for i in range(gran):
                    mag = lower + step * (i + 1)
                    mag = int(mag) if discrete else mag
                    if last_mag is None or mag != last_mag:
                        space.append((op, mag))
                        last_mag = mag
                        
            mags.append(len(space) - pre_space_len)
                
        cls.dims.colsha = len(space)
        cls.COLSHA_SPACE = space
        cls.COLSHA_CAT = len(list(scheme.items()))
        cls.COLSHA_MAG = mags
        
    def __init__(self, num_class, **dists):
        super().__init__()
        self.dists = dists
        self.nclass = num_class
        
    def forward(self, imgs, tgts):
        augs = []

        device = imgs.device

        sampled = Dict({k : dist.sample() for k, dist in self.dists.items()})

        for i, img in enumerate(imgs):
            
            if sampled.flip != {} and sampled.flip[i] == 1:
                img = F.hflip(img)

            if sampled.crop[i] != 0:
                img = cropshift(img, sampled.crop[i].item())

            op, mag = self.COLSHA_SPACE[sampled.colsha[i].item()]
            img = colorshape(img, op, mag)
            
            if not isinstance(img, tc.Tensor):
                img = F.to_tensor(img).to(device)

            if sampled.erase[i] != 0:
                img = erase(img, sampled.erase[i].item(), self.dims.erase)

            augs.append(img)
            
        augs = tc.stack(augs).to(imgs.device, non_blocking=True)

        if sampled.cutmix != {}:
            rand_idx = tc.randperm(augs.size(0)).to(imgs.device, non_blocking=True)
            _augs, _tgts = augs[rand_idx], tgts[rand_idx]

            mix_tgts = []
            
            for aug, _aug, tgt, _tgt, lam in zip(augs, _augs, tgts, _tgts, sampled.cutmix):
                lam = np.float64( lam.item() / (self.dims.cutmix-1))

                if lam != 1.0:
                    lam += np.random.rand() * 1/self.dims.cutmix

                    bbx1, bby1, bbx2, bby2 = rand_bbox(aug.size(), lam)
                    aug[:, bbx1:bbx2, bby1:bby2] = _aug[:, bbx1:bbx2, bby1:bby2]

                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (aug.size()[-1] * aug.size()[-2]))
                tgt = one_hot(tgt, self.nclass)*lam + one_hot(_tgt, self.nclass)*(1-lam)
                mix_tgts.append(tgt)

            tgts = tc.stack(mix_tgts).to(imgs.device, non_blocking=True)
            
        return augs, tgts, sampled
    
    def div_prior_m(self, logits, lower, upper):
        probs = softmax(logits, dim=1)
        logprobs = log_softmax(logits, dim=1)

        uniform_mean = 1 / logits.size(1)
        loss = 0
        count = 0

        if upper is not None:
            upper_dva_idx = probs > uniform_mean * (1 + upper)
            loss += logprobs[upper_dva_idx].sum()
            count += upper_dva_idx.sum()

        if lower is not None:
            lower_dva_idx = probs < uniform_mean * (1 - lower)
            loss -= logprobs[lower_dva_idx].sum()
            count += lower_dva_idx.sum()

        loss = loss if count == 0 else loss / count.detach().clone()
        return loss 

    def div_prior_category_m(self, logits, lower, upper):
        probs = softmax(logits, dim=1)
        logprobs = log_softmax(logits, dim=1)

        cat_mean = 1.0 / len(self.COLSHA_MAG)
        cat_lower = lower if lower is None else cat_mean * (1 - lower)
        cat_upper = upper if upper is None else cat_mean * (1 + upper)
        
        loss = 0
        idx = 0
        count = 0
        for str_len in self.COLSHA_MAG:
            prob = probs[:, idx:idx+str_len].sum(dim=1)
            logprob = logprobs[:, idx:idx+str_len]

            if cat_lower is not None:
                sel = prob < cat_lower
                loss -= logprob[sel].sum()
                count += sel.sum() * str_len

            if cat_upper is not None:
                sel = prob > cat_upper
                loss += logprob[sel].sum()
                count += sel.sum() * str_len

            if str_len > 1:
                prob = softmax(logits[:, idx:idx+str_len], dim=1)
                stren_mean = 1.0 / str_len

                if lower is not None:
                    sel = prob < stren_mean * (1 - lower)
                    loss -= logprob[sel].sum()
                    count += sel.sum()

                if upper is not None:
                    sel = prob > stren_mean * (1 + upper)
                    loss += logprob[sel].sum()
                    count += sel.sum()
                
            idx += str_len

        loss = loss if count == 0 else loss / count.detach().clone()

        return loss

    def div_loss_m(self, logits, lower, upper):
        loss = 0
        count = 0
        for k, logit in logits.items():
            prior = self.div_prior_category_m if k == 'colsha' else self.div_prior_m
            loss += prior(logit, lower, upper)
            count += 1
        return loss / count
        

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def sample_top(x, y):
    x = tc.randint(0, x+1, (1,)).item()
    y = tc.randint(0, y+1, (1,)).item()
    return x, y

def cropshift(img, mag):
    w, h = F.get_image_size(img)
    crop_x = tc.randint(0, mag+1, (1,)).item()
    crop_y = mag - crop_x
    crop_w, crop_h = w - crop_x, h - crop_y

    top_x, top_y = sample_top(crop_x, crop_y)

    img = F.crop(img, top_y, top_x, crop_h, crop_w)
    img = F.pad(img, padding=[crop_x, crop_y], fill=0)

    top_x, top_y = sample_top(crop_x, crop_y)

    return F.crop(img, top_y, top_x, h, w)

def colorshape(img, op_name, mag, interpolation=Interpolation.NEAREST):
    if op_name == 'identity':
        img = img
    elif op_name == 'autocontrast':
        if isinstance(img, tc.Tensor):
            img = F.to_pil_image(img)
        img = F.autocontrast(img)
    elif op_name == 'equalize':
        if isinstance(img, tc.Tensor):
            img = F.to_pil_image(img)
        img = F.equalize(img)
    elif op_name == 'color':
        img = F.adjust_saturation(img, mag)
    elif op_name == 'brightness':
        img = F.adjust_brightness(img, mag)
    elif op_name == 'contrast':
        img = F.adjust_contrast(img, mag)
    elif op_name == 'sharpness':
        img = F.adjust_sharpness(img, mag)
    elif op_name == 'shearx':
        # random sign
        if tc.randint(2, (1,)): mag *= -1
        img = F.affine(img,
                       angle=0.0,
                       translate=[0, 0],
                       scale=1.0,
                       shear=[math.degrees(mag), 0.0],
                       interpolation=interpolation,
                       fill=0)
    elif op_name == 'sheary':
        if tc.randint(2, (1,)): mag *= -1
        img = F.affine(img,
                       angle=0.0,
                       translate=[0, 0],
                       scale=1.0,
                       shear=[0.0, math.degrees(mag)],
                       interpolation=interpolation,
                       fill=0)
    elif op_name == 'shear':
        if tc.randint(2, (1,)): mag *= -1
        shear = [0.0, math.degrees(mag)] if tc.randint(2, (1,)) else [math.degrees(mag), 0.0]
        img = F.affine(img,
                       angle=0.0,
                       translate=[0, 0],
                       scale=1.0,
                       shear=shear,
                       interpolation=interpolation,
                       fill=0)    
    elif op_name == "translatex":
        if tc.randint(2, (1,)): mag *= -1
        img = F.affine(img,
                       angle=0.0,
                       translate=[int(mag), 0],
                       scale=1.0,
                       interpolation=interpolation,
                       shear=[0.0, 0.0])
    elif op_name == "translatey":
        if tc.randint(2, (1,)): mag *= -1
        img = F.affine(img,
                       angle=0.0,
                       translate=[0, int(mag)],
                       scale=1.0,
                       interpolation=interpolation,
                       shear=[0.0, 0.0])
    elif op_name == 'rotate':
        if tc.randint(2, (1,)): mag *= -1
        img = F.rotate(img, angle=int(mag), interpolation=interpolation, fill=0)
    elif op_name == "posterize":
        if isinstance(img, tc.Tensor):
            img = F.to_pil_image(img)
        img = F.posterize(img, int(mag))
    elif op_name == "solarize":
        img = F.solarize(img, mag/256)
    elif op_name == "invert":
        img = F.invert(img)
    else:
        raise ValueError("{} is invalid".format(op_name))
    
    return img


ERASERS = {}
SCALE = (0.0, 0.5)


def erase(img, mag, bins):
    if mag not in ERASERS:
        scale = (SCALE[1] - SCALE[0]) * mag / (bins-1) + SCALE[0]
        eraser = T.RandomErasing(p=1, scale=(scale, scale+0.01))
        ERASERS[mag] = eraser
    else:
        eraser = ERASERS[mag]
        
    return eraser(img)



SPACE = Dict()


SPACE.RA = OrderedDict({
    'identity' : None,
    'autocontrast' : None,
    'equalize' : None,
    'rotate' : (0, 30, True),
    'solarize' : (0, 256, True),
    'color' : (0.1, 1.9, False),
    'posterize' : (4, 8, True),
    'contrast' : (0.1, 1.9, False),
    'brightness' : (0.1, 1.9, False),
    'sharpness' : (0.1, 1.9, False),
    'shearx' : (0, 0.3, False),
    'sheary' : (0, 0.3, False),
    'translatex' : (0, 10, True),
    'translatey' : (0, 10, True)
})
