import json, os, shutil
import numpy as np
import torch as tc
from datetime import datetime
from addict import Dict
from filelock import FileLock
from uuid import uuid4

from mergedeep import merge

RECORD = Dict({'model' : ['arch', 'checkpoint', 'depth', 'width', 'patch_size', 'activation', 'pretrained'],
          'dataset' : ['dataset'],
          'training' : ['criterion', 'optim', 'lr', 'batch_size', 'awp_gamma', 'augment',
                        'search_space', 'cutmix', 'search_gran', 'vul_coef', 'advt', 'trades_beta',
                        'policy_update_n', 'policy_adv_iters', 'trajectory_n', 'warm_aug', 'aff_coef',
                        'rescale_rwd', 'rwd_vul', 'rwd_aff', 'policy_backbone', 'std_ref', 'policy_eval',
                        'div_coef', 'div_limits', 'div_loss', 'plr',
                        'extra_data', 'extra_ratio', 'annealing', 'momentum', 'weight_decay', 'swa', 'clip_grad']})

BASE = 36
ID_LENGTH = 4

INIT_LOG_ID = '0000'

def complete_ids(ids):
    while(len(ids) < ID_LENGTH):
        ids = '0' + ids
    return ids

def ids_from_idx(idx):
    try:
        ids = np.base_repr(idx, BASE).lower()
        ids = complete_ids(ids)
        return ids
    except:
        raise Exception("Invalid index: {}".format(idx))

def idx_from_ids(ids):
    try:
        return int(ids, BASE)
    except:
        raise Exception("Invalid id string: {}".format(ids))


def increase_idx(idx):
    idx = idx_from_ids(idx)
    idx += 1
    return ids_from_idx(idx)
    

def save_log(log, log_fpath, lock=None):
    backup_fpath = log_fpath + '~'
    if os.path.isfile(log_fpath):
        shutil.copyfile(log_fpath, backup_fpath)

    with open(log_fpath, 'w') as f:
        json.dump(log, f, indent=4)

    if lock is not None and lock.is_locked:
        lock.release()
        
    if os.path.isfile(backup_fpath):
        os.remove(backup_fpath)


class Logger:
    FILE = 'json'
    
    def __init__(self, root, name):
        self.log_dir =os.path.join(root, name)
        self.log = None
        self.lock = FileLock(os.path.join(root, '{}.lock'.format(name)))
        
    def new(self, info):
        # job id generation if not given
        job_id = str(uuid4()) if info.job_id is None else info.job_id
        
        self.log = Dict({'id' : None,
                         'job_id' : job_id,
                         'abstract' : info.abstract(),
                         'checkpoint' : {},
                         'robustness' : {},
                         'status' : 'normal'})
        
        RECORD.training += ['eps', 'alpha', 'max_iter', 'random_init', 'warm_start']
            
        for head, attrs in RECORD.items():
            for attr in attrs:
                if hasattr(info, attr):
                    self.log[head][attr] = getattr(info, attr)

        self.log.time.create = self.time()

    def new_id(self):
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
            return ids_from_idx(0)
            
        log_names = os.listdir(self.log_dir)
        log_ids = sorted([idx_from_ids(fname[:-5]) for fname in log_names])

        if len(log_ids) == 0:
            return ids_from_idx(0)

        return ids_from_idx(log_ids[-1] + 1)
        
    def save_train(self):
        if self.log.id is None:
            self.lock.acquire()
            self.log.id = self.new_id()
            
        log_fpath = self.log_fpath_from_id(self.log.id)

        if not os.path.isfile(log_fpath):
            # no log_id file exists
            return save_log(self.log, log_fpath, self.lock)
        
        with open(log_fpath, 'r') as f:
            ext_log = Dict(json.load(f))

        if ext_log.job_id == self.log.job_id:
            # continuous run in the same process
            save_log(self.log, log_fpath, self.lock)
        else:
            # log id is occupied by different runs
            self.log.id = self.new_id()
            self.save_train()
            
    def save_eval(self, log):
        lock_fpath = os.path.join(self.log_dir, '{}.lock'.format(log.id))
        lock = FileLock(lock_fpath)
        log_fpath = os.path.join(self.log_dir, '{}.{}'.format(log.id, self.FILE))
        with lock:
            with open(log_fpath, 'r') as f:
                ext_log = Dict(json.load(f))

            merge(ext_log, log)
            save_log(ext_log, log_fpath)
        os.remove(lock_fpath)
            
    def resume(self, log_id):
        pass
    
    def __getitem__(self, log_id):
        log_fpath = self.log_fpath_from_id(log_id)
        assert os.path.isfile(log_fpath)
        
        with open(log_fpath, 'r') as f:
            log = Dict(json.load(f))
            
        return log

    def log_fpath_from_id(self, log_id):
        return os.path.join(self.log_dir, '{}.{}'.format(log_id, self.FILE))
    
    def time(self):
        return datetime.now().strftime("%d-%m-%Y %H:%M:%S")
