import time
import datetime
import argparse
from options.transformation_options import TransformationOptions
from cyclegan.unaligned_dataset import UnalignedDataset
from cyclegan.cycle_gan_model import CycleGANModel
from visualizer import Visualizer
import torch
import os


# Options
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = TransformationOptions.initialize(parser)
opt = parser.parse_args()
opt_str = ""
exp_name = TransformationOptions.process_opt_str(opt, opt_str)
opt.phase = "train"
opt.isTrain = True
opt.exp_name = exp_name
if opt.debug:
    opt.verbose = True
print("exp-%s" % exp_name)
if not os.path.exists("log"):
    os.makedirs("log")
txt_log_file = "log/%s.txt" % (exp_name)
with open(txt_log_file, 'w') as outfile:
    outfile.write("%s\n" % exp_name)
    outfile.write("%s\n" % str(datetime.datetime.now()))

# GPU
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])

# Visualizer
visualizer = Visualizer(opt)

dataset = UnalignedDataset(opt)
dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))
dataset_size = len(dataloader)
print('The number of training images = %d' % dataset_size)

model = CycleGANModel(opt)
model.setup(opt)    # regular setup: load and print networks; create schedulers
total_iters = 0                # the total number of training iterations
# model.save_networks('latest')
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    for i, data in enumerate(dataloader):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if opt.debug:
            #losses = model.get_current_losses()
            visuals = model.get_current_visuals()
            #print(losses)
            print(model.panel_tracker)
        
        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            model.compute_visuals()
            visualizer.display_current_results(total_iters, model.get_current_visuals())

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.plot_current_losses(total_iters, losses)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)
        iter_data_time = time.time()
    
    expert_results = model.get_expert_selection_results()
    if opt.debug:
        print("Epoch panel results:")
        print(expert_results)
    visualizer.plot_items(epoch, expert_results)
    # txt log
    with open(txt_log_file, 'a') as outfile:
        outfile.write("Epoch %d: %s\n" % (epoch, str(expert_results)))

    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    print(model.epoch_panel_tracker)

    if opt.early_stop_active_expert:
        n_active_expert = (model.epoch_panel_tracker > 0).sum()
        if n_active_expert < opt.n_experts - 1:
            with open(txt_log_file, 'a') as outfile:
                outfile.write("Early stop")
            break

    model.end_epoch()