import torch as tc
import os, glob, sys
import numpy as np
from uuid import uuid4
from argparse import ArgumentParser

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "../..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.config.config import Configuration, DATA, PATH, upper, pixel_2_real
from src.config.config import PARSER as SHARED
from src.data.factory import DATASETS

parser = ArgumentParser(parents=[SHARED])

# Training procedure
parser.add_argument('--rfreq', type=int, default=None,
                    help="the frequency of saving the model for later check")
parser.add_argument('--debug', action='store_true',
                    help="enable debug info display")

# Model
archs = ["wresnet", "paresnet", "vit-b", 'swin-s', 'swin-t']

parser.add_argument('-a', '--arch', choices=archs, default='paresnet',
                    help="the neural network architecture to be used")
parser.add_argument('--checkpoint', default=None,
                    help="the checkpoint of model to be loaded from")
parser.add_argument('--width', type=int, default=1,
                    help="the width factor for widening conv layers")
parser.add_argument('--depth', type=int, default=18,
                    help="depth of wide resnet")
parser.add_argument('-ps', '--patch_size', type=int, default=4,
                    help="depth of wide resnet")
parser.add_argument('--pretrained', action='store_true',
                    help="")
parser.add_argument('-act', '--activation', choices=['relu', 'softplus', 'silu'], default='relu',
                    help="nonlinear activation function")


# Dataset
parser.add_argument('-d', '--dataset', choices=list(DATASETS.keys()), type=upper, default='CIFAR10',
                    help="the dataset to be used")
parser.add_argument('-aug', '--augment',
                    choices=['rcrop', 'auto', 'aa', 'ta', 'uniform', 'cutout',
                             'idbh-c10s', 'idbh-c10w', 'idbh-svhn'],
                    default=None,
                    help="")
parser.add_argument('--download', action='store_true',
                    help="download dataset if not exists")

# Learning Arguments
parser.add_argument('--lr', type=float, default=0.1,
                   help="learning rate in optimizer")
parser.add_argument('--annealing', nargs='+', default=[100, 150],
                   help="learning rate decay every N epochs")
parser.add_argument('--momentum', type=float, default=0.9,
                   help="momentum in optimizer")
parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4,
                   help="weight decay in optimizer")
parser.add_argument('-e', '--epochs', type=int, default=200,
                   help="maximum amount of epochs to be run")
parser.add_argument('--cache', action='store_true',
                   help="cache raw data in memory")

parser.add_argument('--swa', nargs='+', default=None,
                   help="enable stochastic weight averaging")


parser.add_argument('-c', '--criterion', choices=['crossentropy'], default='crossentropy',
                   help="criterion for computing loss")
parser.add_argument('--optim', choices=['sgd', 'adam'], default='sgd',
                   help="optimizer")
parser.add_argument('--nesterov', action='store_true',
                   help="enable nesterov momentum for the optimizer")
parser.add_argument('--clip_grad', action='store_true',
                   help="gradient norm clip for the target model")

'''
Adversray Training

'''

parser.add_argument('--advt', choices=['pgd', 'nfgsm', 'score', 'trades'], default='pgd',
                   help="adv training method")
parser.add_argument('--eps', type=pixel_2_real, default=None,
                    help="attack strength i.e. constraint on the maxmium distortion")
parser.add_argument('--max_iter', type=int, default=10,
                    help="maximum iterations for generating adversary")
parser.add_argument('--alpha', type=pixel_2_real, default=None,
                    help="step size for multi-step attacks")
parser.add_argument('-ri', '--random_init', action='store_true', default=False,
                    help="random iniailization when generating adversarial examples")
parser.add_argument('-ei', '--eval_iter', type=int, default=10,
                    help="number of steps to generate adversary in the main procedure")
parser.add_argument('-ws', '--warm_start', action='store_true', default=False,
                    help="gradually increase perturbation budget in the first 5 epochs")


parser.add_argument('-ed', '--extra-data',
                    choices=['ti80m', 'edm1m', 'edm20m', 'edm50m'],
                    default=None,
                    help="source of extra data")
parser.add_argument('-er', '--extra-ratio', type=float, default=1,
                    help="ratio between original and extra data")

# alternative AT or regularization
parser.add_argument('--trades-beta', type=float, default=None,
                    help="beta in TRADES")
parser.add_argument('-ag', '--awp_gamma', type=float, default=None,
                    help="gamma in AWP")


parser.add_argument('--policy_update_n', type=int, default=5,
                    help="update the policy model for every K iterations")
parser.add_argument('--trajectory_n', type=int, default=8,
                    help="the number of trajectories for REINFORCE, T in the paper")
parser.add_argument('--aff_coef', type=float, nargs='+', default=None,
                    help="the coef of Affinity objective, lambda in the paper")
parser.add_argument('--vul_coef', type=float, nargs='+', default=None,
                    help="the coef of Vulnerability objective")
parser.add_argument('-dl', '--div_limits', nargs=2, type=float, default=[0.5, 2],
                    help="the lower and upper limits for Diversity objective")
parser.add_argument('--div_coef', type=float, default=0.0,
                    help="the coef of Diversity objective, beta in the paper")
parser.add_argument('--div_loss', choices=['mean', 'batchmean'], default='mean',
                    help="the alternatives to compute Diversity")
parser.add_argument('--warm_aug', type=int, default=5,
                    help="applying AROID after N epochs")

parser.add_argument('--policy_adv_iters', type=int, default=2,
                    help="the number of iterations for adversarial generation in Vulnerability")
parser.add_argument('-pb', '--policy_backbone',
                    default='prn18',
                    help="the backbone architecture of the policy model")
parser.add_argument('--std_ref',
                    choices=['wrn34', 'prn18', 'vit-b', 'swin-s', 'swin-t'],
                    default='prn18',
                    help="the Affinity model")
parser.add_argument('--policy_eval', action='store_true', default=False,
                    help="enable eval mode for the policy model")
parser.add_argument('-sg', '--search_gran', type=int, default=10,
                    help="the granularity of search space")
parser.add_argument('--plr', type=float, default=0.001,
                   help="learning rate for updating the policy model")
parser.add_argument('--cutmix',
                    default=None,
                    help="cutmix beta")


class TrainConfig(Configuration):
    def __init__(self):
        super(TrainConfig, self).__init__(parser)
            
        if self.resume is None:
            if self.logging: self.logger.new(self)
        else:
            log = self.logger[self.resume]
            self.logger.log = log

            # resume checkpoint
            self.resume = self.path('trained', "{}/{}_end".format(self.logbook, log.id))
            assert os.path.isfile(self.resume)

            log.status = "resumed"
            
            # resume configuration from log
            configs = {**log.training, **log.model, **log.dataset}
            for k, v in configs.items():
                if hasattr(self, k):
                    default = parser.get_default(k)
                    if getattr(self, k) == default:
                        setattr(self, k, v)

        if self.checkpoint is not None:
            self.checkpoint = self.path('trained', "{}/{}_end".format(self.logbook, self.checkpoint))

        tc.autograd.set_detect_anomaly(self.debug)
        if self.parallel:
            self.batch_size = int(self.batch_size / self.world_size)
        
        if self.swa is not None:
            self.swa_start = int(self.swa[0])
            self.swa_decay = float(self.swa[1]) if self.swa[1] != 'n' else self.swa[1]
            self.swa_freq = int(self.swa[2])

        if self.cutmix is not None and self.cutmix != 'auto':
            self.cutmix = float(self.cutmix)
            
    @property
    def log_required(self):
        return self.logging or self.resume
