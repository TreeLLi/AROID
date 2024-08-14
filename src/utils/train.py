import time, os, signal, copy

import torch as tc
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax, log_softmax, one_hot
from torch.utils.data import DataLoader, Subset
import torch.utils.data.distributed as dd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import detect_anomaly, grad, Variable

import numpy as np
from addict import Dict

from src.data.factory import fetch_dataset, DATASETS
from src.data.dataset import TinyImages, EDM
from src.data.policy import Policy
from src.model.factory import *
from src.model.policy_net import PolicyNet
from src.utils.helper import *
from src.utils.evaluate import validate
from src.utils.printer import sprint, dprint
from src.utils.adversary import pgd, perturb
from src.utils.swa import moving_average, bn_update
from src.utils.awp import AdvWeightPerturb as AWP
from src.utils.score import distance_func, attack_pgd_divergence

def train(args):
    start_epoch = 0
    best_acc1, best_pgd, best_fgsm = 0, 0, 0

    loaders, models, opts = Dict(), Dict(), Dict()

    # training data loaders
    fargs = args.func_arguments(fetch_dataset, DATASETS, postfix='data')
    train_set = fetch_dataset(train=True, **fargs)
    
    train_sampler = dd.DistributedSampler(train_set) if args.parallel else None
    loaders.train = DataLoader(train_set,
                               batch_size=args.batch_size,
                               shuffle=(train_sampler is None),
                               pin_memory=True,
                               num_workers=args.num_workers,
                               sampler=train_sampler,
                               drop_last=True)

    if args.extra_data is not None:
        if args.extra_data == 'ti80m':
            extra_set = TinyImages(fargs.root, train_set.transform, train_set.target_transform)
        elif 'edm' in args.extra_data:
            extra_set = EDM(fargs.root,
                            split=args.extra_data,
                            transform=train_set.transform,
                            target_transform=train_set.target_transform)
        else:
            raise Exception('unknown source of extra data')
        
        extra_sampler = dd.DistributedSampler(extra_set) if args.parallel else None
        loaders.extra = DataLoader(extra_set,
                                   batch_size=int(args.batch_size*args.extra_ratio),
                                   shuffle=(train_sampler is None),
                                   pin_memory=True,
                                   num_workers=args.num_workers,
                                   sampler=extra_sampler,
                                   drop_last=True)

    # test data loaders
    fargs['augment'] = None
    test_set = fetch_dataset(train=False, split=args.world_size, **fargs)
    if args.world_size > 1:
        total_samples = sum([len(vs) for vs in test_set])
        test_set = test_set[args.rank]
    loaders.test = DataLoader(test_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    
    # init policy model
    Policy.init_colsha_space_a('RA', args.search_gran)

    if args.dataset == 'SVHN':
        Policy.dims.pop('flip')

    if args.cutmix in ['auto']:
        Policy.dims['cutmix'] = 11
        
    models.policy = PolicyNet(args.policy_backbone,
                              Policy.dims,
                              input_dim=args.input_dim,
                              patch_size=args.patch_size).to(args.device)
    
    opts.policy = tc.optim.SGD(models.policy.parameters(),
                               lr=args.plr, momentum=args.momentum)
    
    if args.resume is None:
        checkpoint = None
    else:
        checkpoint = tc.load(args.resume, map_location='cpu')
        best_acc1 = checkpoint['best_acc1']
        best_fgsm = checkpoint['best_fgsm']
        best_pgd = checkpoint['best_pgd']
        start_epoch = checkpoint['epoch']
        
        if 'policy_net' in checkpoint:
            models.policy.load_state_dict(checkpoint['policy_net'])
            opts.policy.load_state_dict(checkpoint['policy_opt'])
            
        checkpoint = (args.resume, checkpoint)
        
    models.policy.train()
    
    fargs = args.func_arguments(fetch_model, ARCHS, postfix='arch')        
    if checkpoint is not None:
        fargs['checkpoint'] = checkpoint
    models.target = fetch_model(**fargs).to(args.device)
    
    if args.parallel:
        if args.using_cpu():
            models.target = DDP(models.target)
        else:
            models.target = DDP(models.target, device_ids=[args.rank], output_device=args.rank)

    if args.std_ref == 'prn18':
        models.std = fetch_model('paresnet',
                                 checkpoint=args.path('model', 'std/{}-prn18.pth.tar'.format(args.dataset)),
                                 depth=18,
                                 out_dim=args.out_dim).to(args.device)
    elif args.std_ref == 'wrn34':
        models.std = fetch_model('wresnet',
                                 checkpoint=args.path('model', 'std/{}-wrn34.pth.tar'.format(args.dataset)),
                                 depth=34,
                                 width=10,
                                 out_dim=args.out_dim).to(args.device)
    elif 'swin' in args.std_ref:
        models.std = fetch_model(args.std_ref,
                                 checkpoint=args.path('model', 'std/{}-{}.pth.tar'.format(args.dataset, args.std_ref)),
                                 patch_size=4,
                                 input_dim=args.input_dim,
                                 out_dim=args.out_dim).to(args.device)
    elif 'vit' in args.std_ref:
        models.std = fetch_model(args.std_ref,
                                 checkpoint=args.path('model', 'std/{}-{}.pth.tar'.format(args.dataset, args.std_ref)),
                                 patch_size=args.patch_size,
                                 input_dim=args.input_dim,
                                 out_dim=args.out_dim).to(args.device)
    
    models.std.eval()
    
    if args.awp_gamma is not None:
        awp_proxy = fetch_model(**fargs).to(args.device)
        awp_opt = tc.optim.SGD(awp_proxy.parameters(), lr=0.01)
        awp = AWP(model=model, proxy=awp_proxy, proxy_optim=awp_opt, gamma=args.awp_gamma)
    else:
        awp = None
        
    if args.swa is not None:
        if args.resume is None or start_epoch <= args.swa_start:
            models.swa = fetch_model(**fargs).to(args.device)
            swa_best_acc = 0.0
            swa_best_fgm = 0.0
            swa_best_pgd = 0.0
            args.swa_n = 0
        else:
            swa_ckp = args.resume.replace('_end', '_swa_end') 
            swa_ckp = tc.load(swa_ckp, map_location='cpu')
            fargs['checkpoint'] = (args.resume, swa_ckp)
            models.swa = fetch_model(**fargs).to(args.device)
            swa_best_acc = swa_ckp['best_acc']
            swa_best_fgm = swa_ckp['best_fgm']
            swa_best_pgd = swa_ckp['best_pgd']
            args.swa_n = swa_ckp['num']

        args.swa_freq = len(loaders.train) if args.swa_freq == -1 else args.swa_freq
    
    fargs = args.func_arguments(fetch_criterion, CRITERIA, postfix='grad')
    criterion = fetch_criterion(**fargs)

    fargs = args.func_arguments(fetch_optimizer, OPTIMS, checkpoint=checkpoint)
    opts.target = fetch_optimizer(params=models.target.parameters(), **fargs)
    
    # free the memory taken by the checkpoint
    checkpoint = None
    
    dprint('adversary', **{k:getattr(args, k, None)
                           for k in ['eps', 'alpha', 'max_iter', 'random_init', 'eval_iter']})
    dprint('data loader', batch_size=args.batch_size, num_workers=args.num_workers)
    sprint("=> Start training!", split=True)

    for epoch in range(start_epoch, args.epochs):
        if args.parallel:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(opts.target, epoch, args.lr, args.annealing, args)

        update_dual_models(epoch, loaders, models, opts, criterion, args)
        
        models.target.eval()
        acc1, ig, fgsm, pgd = validate(loaders.test, models.target, criterion, args)
        
        if args.rank is not None and args.rank != 0: continue
        # execute only on the main process
        
        best_acc1 = max(acc1, best_acc1)
        best_fgsm = max(fgsm, best_fgsm)
        best_pgd = max(pgd, best_pgd)

        print(" ** Acc@1: {:.2f} | FGSM: {:.2f} | PGD: {:.2f}".format(best_acc1, best_fgsm, best_pgd))
        
        if args.logging:
            logger = args.logger
            log = logger.log
            
            acc_info = '{:3.2f} E: {} IG: {:.2e} FGSM: {:3.2f} PGD: {:3.2f}'.format(acc1, epoch+1, ig, fgsm, pgd)
            log.checkpoint.end = acc_info
            
            state_dict = models.target.module.state_dict() if args.parallel else models.target.state_dict()
            state = {
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_acc1': best_acc1,
                'best_pgd': best_pgd,
                'best_fgsm' : best_fgsm,
                'optimizer' : opts.target.state_dict(),
                'policy_net' : models.policy.state_dict(),
                'policy_opt' : opts.policy.state_dict()
            }

            if acc1 >= best_acc1:
                log.checkpoint.acc = acc_info

            if pgd >= best_pgd:
                log.checkpoint.pgd = acc_info

            log.time.checkpoint = logger.time()            
            logger.save_train()
            
            fname = "{}/{}".format(args.logbook, log.id)
            ck_path = args.path('trained', fname+"_end")
            tc.save(state, ck_path)
            
            if pgd >= best_pgd:
                shutil.copyfile(ck_path, args.path('trained', fname+'_pgd'))
                                
            if args.swa is not None and args.swa_start <= epoch:
                print(" *  averaging the model")
                t = time.time()
                bn_update(loaders, models, args)
                if args.stage > 0:
                    models.swa.eval()
                    swa_acc, swa_ig, swa_fgm, swa_pgd = validate(loaders.test, models.swa, criterion, args)

                    swa_best_acc = max(swa_acc, swa_best_acc)
                    swa_best_fgm = max(swa_fgm, swa_best_fgm)
                    swa_best_pgd = max(swa_pgd, swa_best_pgd)

                    print(" ** Acc@1: {:.2f} | FGSM: {:.2f} | PGD: {:.2f}".format(swa_best_acc,
                                                                                  swa_best_fgm,
                                                                                  swa_best_pgd))
                    
                state = {'state_dict' : models.swa.state_dict(),
                         'num': args.swa_n,
                         'best_acc' : swa_best_acc,
                         'best_pgd' : swa_best_pgd,
                         'best_fgm' : swa_best_fgm}

                ck_path = args.path('trained', fname+"_swa_end")
                tc.save(state, ck_path)

                if args.stage > 0 and swa_pgd >= swa_best_pgd:
                    shutil.copyfile(ck_path, args.path('trained', fname+'_swa_pgd'))

            if args.rfreq is not None and (epoch+1) % args.rfreq == 0:
                state = models.policy.module.state_dict() if args.parallel else models.policy.state_dict()
                filename = '{}/{}'.format(args.logger.log.id, epoch+1)
                path = args.path('analysis', filename)
                tc.save(state, path)
                    
'''
update two models

'''

def update_dual_models(epoch, loaders, models, opts, criterion, args, **kwargs):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    igs = AverageMeter('IG', ':.2e')
    accs = AverageMeter('Acc', ':6.2f')
    robs = AverageMeter('Rob', ':6.2f')
    
    progress = ProgressMeter(len(loaders.train),
                             [batch_time, losses, igs, accs, robs],
                             prefix="Epoch: [{}]".format(epoch))

    # training adversary config
    if args.warm_start and epoch < 5:
        factor = epoch / 5
        eps = args.eps * factor
        alpha = args.alpha * factor
    else:
        eps, alpha = args.eps, args.alpha
    iters = args.max_iter
    
    models.target.train()
    
    end = time.time()
    niter = len(loaders.train)
    extra_loader = iter(loaders.extra) if 'extra' in loaders else None
    
    for i, (imgs, tgts) in enumerate(loaders.train, 1):
        if extra_loader is not None:
            eimgs, etgts = next(extra_loader)
            imgs = tc.cat((imgs, eimgs))
            tgts = tc.cat((tgts, etgts))

        imgs = imgs.to(args.device, non_blocking=True)
        tgts = tgts.to(args.device, non_blocking=True)

        batch_size = imgs.size(0)
                
        if args.augment=='auto' and epoch >= args.warm_aug:
            if i % args.policy_update_n == 0:
                # update policy network every N iterations
                update_policy_model(loaders, models, opts, criterion, args)

            if args.policy_eval: models.policy.eval()
                
            with tc.inference_mode():
                dists = models.policy.sampler(models.policy(imgs))
                policy = Policy(args.out_dim, **dists)
                
            imgs, tgts, sampled = policy(imgs, tgts)

        acc, rob, loss, ig = update_target_model(imgs,
                                                 tgts,
                                                 models.target,
                                                 opts.target,
                                                 criterion,
                                                 eps,
                                                 alpha,
                                                 iters,
                                                 args)
        
        # log batch statistics to meter
        accs.update(acc, batch_size)
        robs.update(rob, batch_size)
        losses.update(loss, batch_size)
        igs.update(ig, batch_size)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank != 0: continue
        
        if i == 1 or i % args.log_pbtc == 0:
            progress.display(i)

            if args.augment=='auto' and epoch >= args.warm_aug:

                idx = tc.randint(imgs.size(0), (1,)).item()
                probs = {k: dist.probs[idx] for k, dist in dists.items()}
                s = ''
                for k in Policy.dims.keys():
                    if k not in probs: continue
                    prob = probs[k]
                    s += '{}: {}({:.3f})-{}({:.3f})    '.format(k,
                                                                tc.argmax(prob).item(),
                                                                tc.max(prob).item(),
                                                                tc.argmin(prob).item(),
                                                                tc.min(prob).item())
                print(s)
                
        if args.swa is not None and args.swa_start <= epoch and i % args.swa_freq == 0:
            if isinstance(args.swa_decay, str):
                moving_average(models.swa, models.target, 1.0 / (args.swa_n + 1))
                args.swa_n += 1
            else:
                if epoch == args.swa_start and i // args.swa_freq == 1:
                    state_dict = models.target.module.state_dict() if args.parallel else models.target.state_dict()
                    models.swa.load_state_dict(state_dict)
                moving_average(models.swa, models.target, args.swa_decay)

def update_target_model(imgs, tgts, model, opt, criterion, eps, alpha, iters, args):
    prt = tc.zeros_like(imgs)
    if args.random_init: prt.uniform_(-eps, eps)

    cle = tc.clamp(imgs+prt, 0, 1).requires_grad_(True)
    prt = cle - imgs
    lgt_cle = model(cle)
    loss_cle = cross_entropy(lgt_cle, tgts, reduction='none')

    if iters == 0:
        loss = loss_cle.mean()
        lgt_adv = tc.zeros_like(lgt_cle)
        ig = tc.zeros_like(cle)
    else:
        if args.advt is None or args.advt == 'pgd':
            ig = grad(loss_cle.mean(), cle)[0]
            adv, prt = pgd(imgs, tgts, model, criterion, eps, alpha, iters, prt, ig)

            lgt_adv = model(adv)
            loss_adv = cross_entropy(lgt_adv, tgts, reduction='none')
            loss = loss_adv.mean()
        elif args.advt == 'nfgsm':
            # Initialize random step
            unif = 2
            eta = tc.zeros_like(cle).cuda()
            eta.uniform_(-unif * eps, unif * eps)
            eta.requires_grad = True
            
            output = model(tc.clamp(cle + eta, 0, 1))
            loss = criterion(output, tgts)
            ig = tc.autograd.grad(loss, eta)[0]
            ig = ig.detach()
            # Compute perturbation based on sign of gradient
            delta = eta + alpha * tc.sign(ig)
            
            adv = tc.clamp(cle + delta, 0, 1).clone().detach()
            
            lgt_adv = model(adv)
            loss_adv = cross_entropy(lgt_adv, tgts, reduction='none')
            loss = loss_adv.mean()

        elif args.advt == 'trades':
            criterion_kl = nn.KLDivLoss(reduction='sum')
            model.eval()
            batch_size = len(cle)
            # generate adversarial example
            x_adv = cle.detach() + 0.001 * tc.randn(cle.shape).cuda().detach()
            for _ in range(iters):
                x_adv.requires_grad_()
                with tc.enable_grad():
                    loss_kl = criterion_kl(log_softmax(model(x_adv), dim=1),
                                           softmax(model(cle), dim=1))
                ig = tc.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + alpha * tc.sign(ig.detach())
                x_adv = tc.min(tc.max(x_adv, cle - eps), cle + eps)
                x_adv = tc.clamp(x_adv, 0.0, 1.0)
            model.train()

            x_adv = Variable(tc.clamp(x_adv, 0.0, 1.0), requires_grad=False)
            
            lgt_adv = model(x_adv)
            
            # calculate robust loss
            loss_robust = (1.0 / batch_size) * criterion_kl(log_softmax(lgt_adv, dim=1),
                                                            softmax(model(cle), dim=1))
            loss = loss_cle.mean() + args.trades_beta * loss_robust
        elif args.advt == 'score':
            cle = cle.detach()
            delta = attack_pgd_divergence(model, cle, tgts, eps, alpha, 
                                          iters, 1, 'linf', 'LSE', num_classes=args.out_dim)
            
            adv = tc.clamp(cle + delta, min=0, max=1)
            lgt_adv = model(adv)
            robust_output = softmax(lgt_adv, dim=1) # logits

            loss = distance_func(robust_output,
                                 one_hot(tgts, num_classes=args.out_dim).float(),
                                 'LSE').mean()
            ig = tc.ones_like(cle)
            
    opt.zero_grad()
    loss.backward()
    if args.clip_grad:
        tc.nn.utils.clip_grad_norm_(model.parameters(), 1)
    opt.step()

    if tgts.dim() > 1:
        tgts = tc.argmax(tgts, dim=1)
    
    # measure accuracy
    acc = accuracy(lgt_cle, tgts, topk=(1,))[0]
    rob = accuracy(lgt_adv, tgts, topk=(1,))[0]

    ig = tc.norm(ig, p=1)

    opt.zero_grad(set_to_none=True)
    
    return acc.item(), rob.item(), loss.item(), ig.item()


def update_policy_model(loaders, models, opts, criterion, args):
    models.policy.train()
    
    iters = args.policy_adv_iters
    eps = args.eps
    alpha = eps if iters == 1 else eps / 4.0
    
    # for each trajectory, use the same origianl images
    imgs, tgts = next(iter(loaders.train))
    if 'extra' in loaders:
        eimgs, etgts = next(iter(loaders.extra))
        imgs = tc.cat((imgs, eimgs))
        tgts = tc.cat((tgts, etgts))

    if args.dataset == 'INTE':
        # reduce the batch size for Imagenette dataset
        # since our GPU doesn't have enough memory to run the whole batch
        # feel free to remove this reduction if your GPU has enough memory
        imgs = imgs[:100]
        tgts = tgts[:100]
    
    imgs = imgs.to(args.device, non_blocking=True)
    tgts = tgts.to(args.device, non_blocking=True)
    
    logits = models.policy(imgs)
    dists = models.policy.sampler(logits)
    policy = Policy(args.out_dim, **dists)

    r_vuls, r_affs, logprobs = [], [], []

    aff_coef = args.aff_coef[args.stage] if args.stage < len(args.aff_coef) else args.aff_coef[-1]
    if args.vul_coef is not None:
        vul_coef = args.vul_coef[args.stage] if args.stage < len(args.vul_coef) else args.vul_coef[-1]
    else:
        vul_coef = 1
    
    models.target.eval()
    for i in range(args.trajectory_n):
        logprob = 0

        augs, _tgts, sampled = policy(imgs, tgts)

        augs.requires_grad_(True)
        augs_cle_loss = cross_entropy(models.target(augs), _tgts, reduction='none')
        ig = grad(augs_cle_loss.mean(), augs)[0]
        augs_adv, _ = pgd(augs, _tgts, models.target, criterion, eps, alpha, iters, ig=ig)

        with tc.no_grad():
            augs_adv_loss = cross_entropy(models.target(augs_adv), _tgts, reduction='none')            
            augs_std_loss = cross_entropy(models.std(augs), _tgts, reduction='none')
            
        r_vul = augs_adv_loss - augs_cle_loss
        r_aff = augs_std_loss

        with tc.no_grad():
            imgs_std_loss = cross_entropy(models.std(imgs), tgts, reduction='none')
        r_aff -= imgs_std_loss
        r_aff[r_aff<0] = 0

        r_vuls.append(r_vul.detach().clone())
        r_affs.append(r_aff.detach().clone())
        
        for k, dist in dists.items():
            logprob += dist.log_prob(sampled[k])
        
        # trajectory log probability
        logprobs.append(logprob)

        del augs_adv, augs, augs_std_loss, augs_cle_loss, augs_adv_loss, imgs_std_loss, ig

    logprobs = tc.stack(logprobs, dim=1)

    r_vuls = tc.stack(r_vuls, dim=1)
    r_affs = tc.stack(r_affs, dim=1)

    rewards = r_vuls * vul_coef - r_affs * aff_coef
    rewards -= tc.mean(rewards, dim=1, keepdim=True)

    loss = -(logprobs * rewards.detach()).mean()
    
    if args.div_coef != 0:
        div_loss = policy.div_loss_m(logits, *args.div_limits)
        if div_loss is not None:
            loss += div_loss * args.div_coef
            
    opts.policy.zero_grad()
    loss.backward()
    tc.nn.utils.clip_grad_norm_(models.policy.parameters(), 1)
    opts.policy.step()

    opts.policy.zero_grad(set_to_none=True)
    opts.target.zero_grad(set_to_none=True)
    
    models.target.train()

    
def adjust_learning_rate(optimizer, epoch, lr, annealing, args):
    decay = 0
    for a in annealing:
        if epoch < int(a): break
        else: decay += 1
    
    lr *= 0.1 ** decay

    args.stage = decay
    
    params = optimizer.param_groups
    if lr != params[0]['lr']:
        sprint("Learning rate now is {:.0e}".format(lr))
    
    for param in params: param['lr'] = lr

