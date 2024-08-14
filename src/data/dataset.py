import os
import torch as tc
import numpy as np
import pickle

import torchvision.datasets as datasets
from torchvision.transforms import functional as F
from torch.utils.data import Dataset

from PIL import Image
    
class SVHN(datasets.SVHN):
    PATH = 'SVHN'
    
    def __init__(self, root, train, transform=None, tgt_transform=None, cache=False, download=False):
        split = 'train' if train else 'test'
        root = os.path.join(root, self.PATH)
        super().__init__(root, split, transform, tgt_transform, download)

        self.cache = cache

    
class TinyImageNet(datasets.ImageFolder):
    PATH = 'tiny-imagenet-200'
    
    def __init__(self, root, train=True, cache=False, transform=None, tgt_transform=None, download=False):
        path = os.path.join(self.PATH, 'train' if train else 'val')
        root = os.path.join(root, path)
        super().__init__(root, transform, tgt_transform)

        # TODO: download
        
        if cache:
            self.cache = {}
        else:
            self.cache = None
            
    def __getitem__(self, idx):
        if self.cache is None:
            return self.get_item(idx)
        
        if idx in self.cache:
            img, tgt = self.cache[idx]
            return augment(img, tgt, self.transform, self.tgt_transform)
            
        img, aug, tgt = self.get_item(idx)
        self.cache[idx] = (img, tgt)
        return img, aug, tgt

    def get_item(self, idx):
        path, tgt = self.samples[idx]
        img = self.loader(path)

        aug = img if self.transform is None else self.transform(img)
        tgt = tgt if self.tgt_transform is None else self.tgt_transform(tgt)
        
        return img, aug, tgt

class Imagenette(datasets.ImageFolder):
    PATH = 'imagenette'
    
    def __init__(self, root, train=True, cache=False, transform=None, tgt_transform=None, download=False):
        path = os.path.join(self.PATH, 'train' if train else 'val')
        root = os.path.join(root, path)
        super().__init__(root, transform, tgt_transform)


    
class TinyImages(Dataset):
    DATA_FILENAME = 'ti_500K_pseudo_labeled.pickle'
    
    def __init__(self, root, transform=None, tgt_transform=None, download=False):
        super().__init__()

        data_path = os.path.join(root, self.DATA_FILENAME)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data = data['data']
        self.tgts = data['extrapolated_targets']

        self.transform = transform
        self.tgt_transform = tgt_transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img, tgt = self.data[idx], self.tgts[idx]
        img = Image.fromarray(img)

        aug = img if self.transform is None else self.transform(img)
        tgt = tgt if self.tgt_transform is None else self.tgt_transform(tgt)
        
        return aug, tgt


class EDM(Dataset):
    def __init__(self, root, split='edm50m', transform=None, target_transform=None):
        super().__init__()

        split = split.replace('edm', '')
        self.data_path = os.path.join(root, f'edm_synt/CIFAR10/{split}.npz')
        data = np.load(self.data_path)

        self.transform = transform
        self.target_transform = target_transform

        self.imgs, self.tgts = data['image'], data['label']
        
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img, tgt = self.imgs[idx], self.tgts[idx]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            tgt = self.target_transform(tgt)

        return img, tgt
