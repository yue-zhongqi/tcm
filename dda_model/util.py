import torch
from torch.nn import init
import os
import numpy as np
import random


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[], parallel=True, init_weight=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if parallel:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_weight:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def get_cdm_path(cdm_root, exp_name, dataset_name, self_name, other_domain_name, visualize=False):
    if visualize:
        save_path = os.path.join(cdm_root, "visualize", exp_name, dataset_name)
    else:
        save_path = os.path.join(cdm_root, exp_name, dataset_name)
    cdm_path = os.path.join(save_path, "%s2%s" % (self_name, other_domain_name))
    return cdm_path

def get_cdm_file_name(path):
    class_name, img_name = path.split('/')[-2:]
    return "%s_%s.png" % (class_name, img_name)

def get_expert_cdm_file_name(path, expert_idx):
    class_name, img_name = path.split('/')[-2:]
    return "%s_%s_%d.png" % (class_name, img_name, expert_idx)

def scale_to_tensor(tensor):
    # scale to 0->1
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True