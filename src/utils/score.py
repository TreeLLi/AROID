import torch
import torch.nn.functional as F
import torch.nn as nn

criterion_kl = nn.KLDivLoss(reduction='none')
upper_limit, lower_limit = 1, 0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def distance_func(output, target, divergence): # both output and target are summed to 1
    M = (output + target) / 2
    if divergence == 'JSsqrt':
        return (0.5 * (criterion_kl(M.log(), output) + criterion_kl(M.log(), target))).sum(dim=-1).sqrt()
    elif divergence == 'JS':
        return (0.5 * (criterion_kl(M.log(), output) + criterion_kl(M.log(), target))).sum(dim=-1)
    elif divergence == 'LSE':
        return torch.sum((output - target) ** 2, dim=-1)
    elif divergence == 'L1square':
        return (output - target).abs().sum(dim=-1).square()
    elif divergence == 'KL':
        return criterion_kl(output.log(), target).sum(dim=-1)
    elif divergence == 'KLsqrt':
        return criterion_kl(output.log(), target).sum(dim=-1).sqrt()
    elif divergence == 'RKLsqrt':
        return criterion_kl(target.log(), output).sum(dim=-1).sqrt()
    else:
        return torch.norm(output - target, dim=-1, p=float(divergence))


def attack_pgd_divergence(model, X, y, eps, alpha, attack_iters, restarts, norm, divergence='1', num_classes=10):
    bs = y.shape[0]
    y = F.one_hot(y, num_classes=num_classes)
    max_loss = torch.zeros(bs).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "linf":
            delta.uniform_(-eps, eps)
        elif norm == "l2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*eps
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            output = F.softmax(output, dim=1)
            loss = distance_func(output, y.float(), divergence)
            loss.mean().backward()
            grad = delta.grad.detach()
            if norm == "linf":
                delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)
            elif norm == "l2":
                g_norm = torch.norm(grad.view(bs,-1),dim=1).view(bs,1,1,1)
                scaled_g = grad/(g_norm + 1e-10)
                delta.data = (delta + scaled_g*alpha).view(bs,-1).renorm(p=2,dim=0,maxnorm=eps).view_as(delta)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.grad.zero_()
    return delta.detach()
