import argparse
import os
import datetime
import pre_process as prep
from options.dda_options import DDAOptions
from options.transformation_options import TransformationOptions
from data_list import BundledImageList
from dda_model.dataloader import DDALoader
from dda_model.dda_model import DDAModel
from dda_model.dda_model2 import DDAModel2
from dda_model.dda_model3 import DDAModel3
from dda_model.util import get_cdm_path, setup_seed
from cyclegan.cycle_gan_model import CycleGANModel
from visualizer import Visualizer
import torch
import numpy as np

# DDA options
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = DDAOptions.initialize(parser)
# parser = TransformationOptions.initialize(parser)
opt = parser.parse_args()
opt_str = ""
exp_name = DDAOptions.process_opt_str(opt, opt_str)
opt.dda_exp_name = exp_name
opt.isTrain = False     # CycleGAN not training
opt.phase = "test"      # CycleGAN testing
if opt.debug:
    opt.verbose = True
print("Experiment: %s" % exp_name)
if not opt.dda_continue_train:
    if not os.path.exists("log/dda/%s/%s2%s" % (opt.dataset_name, opt.s_name, opt.t_name)):
        os.makedirs("log/dda/%s/%s2%s" % (opt.dataset_name, opt.s_name, opt.t_name))
    txt_log_file = "log/dda/%s/%s2%s/%s.txt" % (opt.dataset_name, opt.s_name, opt.t_name, exp_name)
    with open(txt_log_file, 'w') as outfile:
        outfile.write("%s\n" % exp_name)
        outfile.write("%s\n" % str(datetime.datetime.now()))

# Random seed
setup_seed(2021)

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
source_cdm_path = get_cdm_path(opt.cdm_path, opt.cdt_exp_name, opt.dataset_name, opt.s_name, opt.t_name)
target_cdm_path = get_cdm_path(opt.cdm_path, opt.cdt_exp_name, opt.dataset_name, opt.t_name, opt.s_name)
if opt.bundle_transform:
    img_transform = prep.bt_image(resize_size=256, crop_size=224)
    bundle_transform = prep.bt_bundle()
    if opt.bundle_resized_crop:
        resized_crop_size = 224
    else:
        resized_crop_size = 0
    train_dset = BundledImageList(open(opt.s_dset).readlines(), ori_transform=img_transform, return_path=False,
                                  cdm_path=source_cdm_path, bundled_transform=bundle_transform,
                                  resized_crop_size=resized_crop_size, random_horizontal_flip=True)
    target_dset = BundledImageList(open(opt.t_dset).readlines(), ori_transform=img_transform, return_path=False,
                                   cdm_path=target_cdm_path, bundled_transform=bundle_transform,
                                   resized_crop_size=resized_crop_size, random_horizontal_flip=True)
    test_dset = BundledImageList(open(opt.t_dset).readlines(), ori_transform=img_transform, return_path=False,
                                  cdm_path=target_cdm_path, bundled_transform=bundle_transform,
                                  resized_crop_size=resized_crop_size, random_horizontal_flip=True)
    source_loader = DDALoader(opt, opt.s_dset, opt.train_batch_size, dset=train_dset)
    target_loader = DDALoader(opt, opt.t_dset, opt.train_batch_size, dset=target_dset)
    test_loader = DDALoader(opt, opt.t_dset, opt.train_batch_size, dset=test_dset)
else:
    train_transform = prep.image_target(resize_size=256, crop_size=224, alexnet=False)
    test_transform = prep.image_test(resize_size=256, crop_size=224, alexnet=False)
    cdm_train_transform = prep.cdm_train()
    cdm_test_transform = prep.cdm_test()

    source_loader = DDALoader(opt, opt.s_dset, opt.train_batch_size, noflip=False, transform=train_transform,
                              return_cdm=True, cdm_path=source_cdm_path, cdm_transform=cdm_train_transform)
    target_loader = DDALoader(opt, opt.t_dset, opt.train_batch_size, noflip=False, transform=train_transform,
                              return_cdm=True, cdm_path=target_cdm_path, cdm_transform=cdm_train_transform)
    test_loader = DDALoader(opt, opt.t_dset, opt.test_batch_size, noflip=True, transform=test_transform,
                              return_cdm=True, cdm_path=target_cdm_path, cdm_transform=cdm_test_transform)

# DDA Model
if opt.use_dda2:
    dda = DDAModel2(opt)
elif opt.all_experts:
    dda = DDAModel3(opt)
else:
    dda = DDAModel(opt)
dda.train_mode()
dda.setup()
if opt.all_experts:
    dda.setup_g_mean(source_loader, target_loader)

# Visualizer
visualizer = Visualizer(opt, exp_name)

best_accuracy = 0.0
total_iters = 0
for i in range(opt.n_iterations):
    total_iters += 1

    # Training
    dda.train_mode()
    inputs_source, labels_source, s2t_mappings = source_loader.next()
    inputs_target, _, t2s_mappings = target_loader.next()
    dda.set_input(inputs_source, labels_source, s2t_mappings, inputs_target, t2s_mappings)
    train_accuracy = dda.optimize()

    # Record loss
    if total_iters % opt.dda_print_freq == 0:    # print training losses and save logging information to the disk
        losses = dda.get_current_losses()
        visualizer.plot_current_losses(total_iters, losses)

    # Testing
    if i % opt.test_interval == opt.test_interval - 1:
        dda.test_mode()
        with torch.no_grad():
            results = []
            if opt.accurate_mu:
                dda.target_mu = dda.get_t2s_mu(test_loader)
            for j in range(test_loader.length):
                inputs, labels, test_t2s = test_loader.next()
                dda.set_input(None, None, None, inputs, test_t2s)
                logits = dda.predict()
                _, prediction = torch.max(logits, 1)
                results += (prediction.cpu() == labels).float().detach().numpy().tolist()
        accuracy = np.array(results).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            dda.save_networks("best")
        if not opt.debug:
            visualizer.writer.add_scalar("test_accuracy", accuracy, total_iters)
        log_msg = "Iteration%d: Train Accuracy %.4f Test Accuracy %.4f Best Accuracy %.4f" % (i, train_accuracy, accuracy, best_accuracy)
        print(log_msg)
        with open(txt_log_file, 'a') as outfile:
            outfile.write("%s\n" % (log_msg))
    # Save Network
    if (i % opt.snapshot_interval == 0 and i > 0):
        print('saving the latest model (epoch %d, total_iters %d)' % (total_iters / source_loader.length, total_iters))
        save_suffix = 'iter_%d' % total_iters
        dda.save_networks(save_suffix)