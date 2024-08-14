import torch as tc
import torchvision.transforms as T

from torch.utils.data import random_split, Subset

from src.utils.printer import dprint
from src.data.dataset import *
from src.data.transform import Cutout
from src.data.idbh import IDBH
from src.data.uniform import UniformAROID

from torchvision.datasets import CIFAR10, CIFAR100

DATASETS = {'TIN' : TinyImageNet,
            'CIFAR10' : CIFAR10,
            'CIFAR100' : CIFAR100,
            'SVHN' : SVHN,
            'INTE' : Imagenette}


def fetch_dataset(dataset, root, train, augment=None, input_dim=None, cache=False, split=False,
                  download=False, **config):
    
    assert dataset in DATASETS
    
    # hyper-parameter report
    head = 'Training Set' if train else 'Test Set'
    dprint(head, dataset=dataset, augment=augment, cache=cache, **config)

    t = []
    if dataset == 'INTE':
        t.append(T.Resize(input_dim[1:]))

    if augment == 'rcrop':
        t += [T.RandomHorizontalFlip(),
              T.RandomCrop(input_dim[1:], padding=4),
              T.ToTensor()]
    elif augment is not None and augment[:4] == 'idbh':
        version = augment.split('-')[1]
        t += [IDBH(version)]
    elif augment == 'aa':
        from torchvision.transforms.autoaugment import AutoAugmentPolicy
        
        if dataset in ['CIFAR10', 'CIFAR100']:
            policy = AutoAugmentPolicy.CIFAR10
        elif dataset == 'SVHN':
            policy = AutoAugmentPolicy.SVHN
        else:
            policy = AutoAugmentPolicy.IMAGENET
            
        t += [T.RandomHorizontalFlip(),
              T.RandomCrop(input_dim[1:], padding=4),
              T.AutoAugment(policy),
              T.ToTensor(),
              Cutout(16)]
    elif augment == 'ta':
        t += [T.RandomHorizontalFlip(),
              T.RandomCrop(input_dim[1:], padding=4),
              T.TrivialAugmentWide(),
              T.ToTensor(),
              Cutout(16)]
    elif augment == 'cutout':
        t += [T.RandomHorizontalFlip(),
              T.RandomCrop(input_dim[1:], padding=4),
              T.ToTensor(),
              Cutout(20)]
    elif augment == 'uniform':
        t += [UniformAROID()]
    else:
        t.append(T.ToTensor())
        
    t = T.Compose(t)
        
    dataset = DATASETS[dataset](root, train, transform=t, download=download, **config)

    if split > 1:
        total = len(dataset)
        chunk = total // split
        split = [chunk for i in range(split-1)]
        split = [0] + split + [total-sum(split)]
        split = [sum(split[:i+1]) for i, _ in enumerate(split)]
        indices = list(range(len(dataset)))
        dataset = [Subset(dataset, indices[s:split[i+1]]) for i, s in enumerate(split[:-1])]

    return dataset
