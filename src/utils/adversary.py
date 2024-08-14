import torch as tc
from torch.autograd import grad, Variable

from torchattacks import PGD, APGD, FGSM, Jitter
from autoattack import AutoAttack

from src.utils.printer import dprint

from addict import Dict

def input_grad(imgs, targets, model, criterion):
    output = model(imgs)
    loss = criterion(output, targets)
    ig = grad(loss, imgs)[0]
    return ig

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    adv = imgs.requires_grad_(True) if pert is None else tc.clamp(imgs+pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step*tc.sign(ig)
    else:
        pert += eps_step*tc.sign(ig)
    pert.clamp_(-eps, eps)
    adv = tc.clamp(imgs+pert, 0, 1)
    pert = adv-imgs
    return adv.detach(), pert.detach()

def pgd(imgs, targets, model, criterion, eps, eps_step, max_iter, pert=None, ig=None):
    for i in range(max_iter):
        adv, pert = perturb(imgs, targets, model, criterion, eps, eps_step, pert, ig)
        ig = None
    return adv, pert

def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = tc.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - tc.clamp(real - other + confidence, min=0.)
    loss = tc.sum(loss)
    return loss

ATTACK = {
    'FGM' : FGSM,
    'PGD' : PGD,
    'APGD' : APGD,
    'AA' : AutoAttack,
    'AA+' : AutoAttack,
    'AA--' : AutoAttack,
    'CW' : pgd,
    'JITTER' : Jitter
}

def fetch_attack(attack, model, **config):
    
    config = Dict(config)
    
    if attack == 'AA':
        config.version = 'standard'
    elif attack == 'AA+':
        config.version = 'plus'
    elif attack == 'AA--':
        config.version = 'custom'
        config.attacks_to_run = ['apgd-ce', 'apgd-dlr']
        
    if 'seed' in config and config['seed'] is None:
        config['seed'] = 0

    return ATTACK[attack](model, **config)
    

class Attacker:

    def __init__(self, attack, model, norm, eps, batch_size, **kwargs):
        kwargs = Dict(kwargs)
        self.attack = attack
        if attack == 'AA--':
            self.attacker = AutoAttack(model,
                                       norm=norm,
                                       eps=eps,
                                       version='custom',
                                       verbose=False,
                                       attacks_to_run=['apgd-ce', 'apgd-t'])
            self.attacker.apgd.n_restarts = 1
            self.batch_size = batch_size
        elif attack == 'AA':
            self.attacker = AutoAttack(model,
                                       norm=norm,
                                       eps=eps,
                                       version='standard',
                                       verbose=False)
            self.batch_size = batch_size
        elif attack == 'AA+':
            self.attacker = AutoAttack(model,
                                       norm=norm,
                                       eps=eps,
                                       version='plus',
                                       verbose=False)
            self.batch_size = batch_size
        elif attack == 'JITTER':
            self.attacker = Jitter(model, eps=eps, alpha=kwargs.alpha, steps=kwargs.steps)
        elif attack == 'PGD':
            self.attacker = PGD(model,
                                eps=eps,
                                alpha=kwargs.alpha,
                                steps=kwargs.steps,
                                random_start=kwargs.n_restart)
            
        self.model = model
        self.norm = norm
        self.eps = eps
        self.alpha = kwargs.alpha
        self.n_restart = kwargs.n_restart
        self.steps = kwargs.steps
        
    def perturb(self, x, y):
        if 'AA' in self.attack:
            adv = self.attacker.run_standard_evaluation(x, y, bs=self.batch_size)
        elif self.attack == 'CW':
            adv, _ = pgd(x, y, self.model, CWLoss, self.eps, self.alpha, self.steps)
        elif self.attack in ['JITTER', 'PGD']:
            adv = self.attacker(x, y)
            
        return adv
