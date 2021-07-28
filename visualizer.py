import numpy as np
import os
import sys
import ntpath
import time
from tensorboardX import SummaryWriter


class Visualizer():
    def __init__(self, opt, exp_name=None):
        if exp_name is None:
            exp_name = opt.exp_name
        if not opt.debug:
            log_dir = "runs/%s" % (exp_name)
            self.writer = SummaryWriter(log_dir)
            self.debug = False
        else:
            self.debug = True

    def plot_current_losses(self, iteration, losses):
        if self.debug:
            return
        for name, value in losses.items():
            self.writer.add_scalar(name, value, iteration)

    def display_current_results(self, iteration, visuals):
        if self.debug:
            return
        for name, value in visuals.items():
            self.writer.add_images(name, value, global_step=iteration, walltime=iteration)

    def plot_items(self, iteration, items):
        if self.debug:
            return
        for name, value in items.items():
            self.writer.add_scalar(name, value, iteration)