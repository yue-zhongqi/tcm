# This file generates cross-domain mappings for source and target domains

import argparse
import torch
import os
from torchvision.utils import save_image
from options.transformation_options import TransformationOptions
from options.cdm_options import CDMOptions
from dda_model.dataloader import DDALoader
from cyclegan.cycle_gan_model import CycleGANModel
from dda_model.util import get_cdm_path, get_cdm_file_name, scale_to_tensor, get_expert_cdm_file_name

# Options
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = TransformationOptions.initialize(parser)
parser = CDMOptions.initialize(parser)
opt = parser.parse_args()
opt_str = ""
opt_str = CDMOptions.process_opt_str(opt, opt_str)
exp_name = TransformationOptions.process_opt_str(opt, opt_str)
opt.continue_train = True   # CycleGAN loads latest
opt.isTrain = True     # CycleGAN needs to load discriminator
opt.phase = "train"      # CycleGAN needs to load discriminator
opt.exp_name = exp_name
if opt.debug:
    opt.verbose = True
print("Generating cross-domain mappings for experiment-%s" % (exp_name))

# CDM save path
print(os.path.join(opt.checkpoints_dir, exp_name))
assert os.path.exists(os.path.join(opt.checkpoints_dir, exp_name))
source_cdm_path = get_cdm_path(opt.cdm_path, exp_name, opt.dataset_name, opt.s_name, opt.t_name, visualize=opt.visualize)
target_cdm_path = get_cdm_path(opt.cdm_path, exp_name, opt.dataset_name, opt.t_name, opt.s_name, visualize=opt.visualize)
if not os.path.exists(source_cdm_path):
    os.makedirs(source_cdm_path)
if not os.path.exists(target_cdm_path):
    os.makedirs(target_cdm_path)

# GPU
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

# Data Loader
source_loader = DDALoader(opt, opt.s_dset, opt.batch_size, noflip=True, return_path=True, drop_last=False, center=True)
target_loader = DDALoader(opt, opt.t_dset, opt.batch_size, noflip=True, return_path=True, drop_last=False, center=True)

# CycleGAN model
cdt = CycleGANModel(opt)
cdt.setup(opt)
cdt.cdm_mode = True

n_s = source_loader.length
n_t = target_loader.length

if opt.visualize:
    for i in range(opt.n_experts):
        if not os.path.exists(os.path.join(source_cdm_path, str(i))):
            os.makedirs(os.path.join(source_cdm_path, str(i)))
        if not os.path.exists(os.path.join(target_cdm_path, str(i))):
            os.makedirs(os.path.join(target_cdm_path, str(i)))

for i in range(max(n_s, n_t)):
    if i < n_s:
        source_imgs, _, source_paths = source_loader.next()
        if source_imgs.shape[0] < opt.batch_size:
            diff = opt.batch_size - source_imgs.shape[0]
            source_imgs_extra, _, source_paths_extra = source_loader.next()
            source_imgs = torch.cat((source_imgs, source_imgs_extra[:diff]), dim=0)
            source_paths = source_paths + source_paths_extra[:diff]
            assert source_imgs.shape[0] == opt.batch_size
    if i < n_t:
        target_imgs, _, target_paths = target_loader.next()
        if target_imgs.shape[0] < opt.batch_size:
            diff = opt.batch_size - target_imgs.shape[0]
            target_imgs_extra, _, target_paths_extra = target_loader.next()
            target_imgs = torch.cat((target_imgs, target_imgs_extra[:diff]), dim=0)
            target_paths = target_paths + target_paths_extra[:diff]
            assert target_imgs.shape[0] == opt.batch_size
    inputs = {'A': source_imgs, 'B': target_imgs, 'A_paths': source_paths, 'B_paths': target_paths}
    cdt.set_input(inputs)
    with torch.no_grad():
        cdt.forward()
        cdt.backward_G(backward_loss=False)
    if i < n_s:
        for j in range(source_imgs.shape[0]):
            if not opt.visualize:
                if not opt.all_experts:
                    cdm_file_path = os.path.join(source_cdm_path, get_cdm_file_name(source_paths[j]))
                    save_image(scale_to_tensor(cdt.fake_B[j]), cdm_file_path)
                else:
                    for k in range(opt.n_experts):
                        cdm_file_path = os.path.join(source_cdm_path, get_expert_cdm_file_name(source_paths[j], k))
                        save_image(scale_to_tensor(cdt.fake_B_all[k][j]), cdm_file_path)
            else:
                if not opt.all_experts:
                    expert_id = cdt.expert_idx[j].item()
                    cdm_file_path = os.path.join(source_cdm_path, str(expert_id), get_cdm_file_name(source_paths[j]))
                    side2side_img = torch.cat((cdt.real_A[j], cdt.fake_B[j]), dim=-1)
                    save_image(scale_to_tensor(side2side_img), cdm_file_path)
                else:
                    for k in range(opt.n_experts):
                        cdm_file_path = os.path.join(source_cdm_path, get_expert_cdm_file_name(source_paths[j], k))
                        side2side_img = torch.cat((cdt.real_A[j], cdt.fake_B_all[k][j]), dim=-1)
                        save_image(scale_to_tensor(side2side_img), cdm_file_path)
    if i < n_t:
        for j in range(target_imgs.shape[0]):
            if not opt.visualize:
                if not opt.all_experts:
                    cdm_file_path = os.path.join(target_cdm_path, get_cdm_file_name(target_paths[j]))
                    save_image(scale_to_tensor(cdt.fake_A[j]), cdm_file_path)
                else:
                    for k in range(opt.n_experts):
                        cdm_file_path = os.path.join(target_cdm_path, get_expert_cdm_file_name(target_paths[j], k))
                        save_image(scale_to_tensor(cdt.fake_A_all[k][j]), cdm_file_path)
            else:
                if not opt.all_experts:
                    expert_id = cdt.expert_idx[j + target_imgs.shape[0]].item()
                    cdm_file_path = os.path.join(target_cdm_path, str(expert_id), get_cdm_file_name(target_paths[j]))
                    side2side_img = torch.cat((cdt.real_B[j], cdt.fake_A[j]), dim=-1)
                    save_image(scale_to_tensor(side2side_img), cdm_file_path)
                else:
                    for k in range(opt.n_experts):
                        cdm_file_path = os.path.join(target_cdm_path, get_expert_cdm_file_name(target_paths[j], k))
                        side2side_img = torch.cat((cdt.real_B[j], cdt.fake_A_all[k][j]), dim=-1)
                        save_image(scale_to_tensor(side2side_img), cdm_file_path)
    print("%d/%d completed." % (i, max(n_s, n_t)))