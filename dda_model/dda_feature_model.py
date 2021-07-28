import torch
import torch.nn as nn
import os
from collections import OrderedDict
from cyclegan.feature_cycle_gan_model import FeatureCycleGANModel
from dda_model.vae import VAE
from dda_model.backbone import ResNet
from dda_model.loss import get_alignment_loss, calc_coeff, NMTCritierion
from dda_model.util import init_net
from network import AdversarialNetwork


class DDAFeatureModel():
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU 
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.exp_name)  # save all the checkpoints to save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Networks
        self.backbone = ResNet(opt.resnet_name, return_feature_map=True)
        self.x_dim = self.backbone.n_features
        self.z_dim = opt.z_dim
        self.num_classes = opt.num_classes
        self.n_experts = opt.n_experts
        if not opt.baseline:
            self.vae = VAE(self.x_dim, self.z_dim)
            self.linear_y = nn.Linear(self.x_dim + self.z_dim, self.num_classes)
            self.linear_tilde_x = nn.Linear(self.x_dim + self.z_dim, self.x_dim)
            self.model_names = ['vae', 'backbone', 'linear_y', 'linear_tilde_x']
            self.loss_names = ["vae", "linear", "causal"]
        else:
            self.linear_y = nn.Linear(self.x_dim, self.num_classes)
            self.model_names = ['backbone', 'linear_y']
            self.loss_names = ['y']
        self.init_networks()
        self.cyclegan = FeatureCycleGANModel(opt, self.x_dim)

        # Optimizers
        if not opt.baseline:
            self.backbone_opt = torch.optim.SGD(self.backbone.parameters(), lr=opt.backbone_lr, momentum=0.9,
                                                weight_decay=0.001, nesterov=True)
            self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=opt.vae_lr, weight_decay=0.001)
            self.linear_opt = torch.optim.SGD(list(self.linear_y.parameters()) + list(self.linear_tilde_x.parameters()),
                                              lr=opt.linear_lr, momentum=opt.linear_momentum, weight_decay=opt.linear_weight_decay, nesterov=True)
            self.optimizer_names = ["backbone", "vae", "linear"]
        else:
            self.linear_opt = torch.optim.SGD(self.linear_y.parameters(), lr=opt.linear_lr,
                                              momentum=opt.linear_momentum, weight_decay=opt.linear_weight_decay, nesterov=True)
            self.backbone_opt = torch.optim.SGD(self.backbone.parameters(), lr=opt.backbone_lr, momentum=0.9, weight_decay=0.001, nesterov=True)
            self.optimizer_names = ["backbone", "linear"]
        self.init_schedulers()
        # Losses
        self.mse = nn.MSELoss()
        if self.opt.label_smoothing:
            self.ce = NMTCritierion(0.1)
        else:
            self.ce = nn.CrossEntropyLoss()
        
        self.iteration = 0

    def init_schedulers(self):
        lambda_lr = lambda iter: (1 + 0.001 * iter) ** (-0.75)
        for name in self.optimizer_names:
            optimizer = getattr(self, name + "_opt")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
            setattr(self, name + "_scheduler", scheduler)

    def step_schedulers(self):
        for name in self.optimizer_names:
            scheduler = getattr(self, name + "_scheduler")
            scheduler.step()

    def init_networks(self):
        for name in self.model_names:
            if isinstance(name, str):
                parallel = (name == "backbone")
                init_weight = (name != "backbone")
                net = init_net(getattr(self, name), self.opt.init_type, self.opt.init_gain, self.gpu_ids, parallel, init_weight)
                setattr(self, name, net)

    def train_mode(self):
        self.train = True
        for name in self.model_names:
            getattr(self, name).train()

    def test_mode(self):
        self.train = False
        for name in self.model_names:
            getattr(self, name).eval()

    def set_input(self, source_imgs, source_labels, target_imgs):
        if source_imgs is not None:
            # During testing, there may be no source images
            self.source_imgs = source_imgs.to(self.device)
        else:
            self.source_imgs = None
        if source_labels is not None:
            # During testing, there may be no source labels
            self.source_labels = source_labels.to(self.device)
        else:
            self.source_labels = None
        self.target_imgs = target_imgs.to(self.device)

    def get_logits_during_train(self, x_s, x_t, x_s2t, x_t2s):
        if self.opt.baseline:
            x = torch.cat((x_s2t, x_t), dim=0)
            logits = self.linear_y(x)
        else:
            b0 = self.linear_y.bias.data.unsqueeze(1)
            w1 = self.linear_y.weight.data[:, :self.z_dim]
            w2 = self.linear_y.weight.data[:, self.z_dim:]
            w3 = self.linear_tilde_x.weight.data[:, :self.z_dim]
            w3i = torch.pinverse(w3)
            w4 = self.linear_tilde_x.weight.data[:, self.z_dim:]
            b1 = self.linear_tilde_x.bias.data.unsqueeze(1)
            if self.opt.use_target_estimate:
                mu_tildeX = self.get_running_mu(x_t2s, False)
            else:
                mu_tildeX = self.get_running_mu(x_s, False)
            mu_xs = mu_tildeX
            mu_xt = mu_tildeX
            w1w3i = torch.mm(w1, w3i)
            w2w1w3w4 = w2 - torch.mm(w1w3i, w4)  # nc * x_dim
            bs = x_s2t.shape[0]
            xs_logits = (b0 - torch.mm(w1w3i, b1) + torch.mm(w1w3i, mu_xs)).expand(-1, bs).t() + torch.mm(w2w1w3w4, x_s2t.t()).t()
            xt_logits = (b0 - torch.mm(w1w3i, b1) + torch.mm(w1w3i, mu_xt)).expand(-1, bs).t() + torch.mm(w2w1w3w4, x_t.t()).t()
            logits = torch.cat((xs_logits, xt_logits), dim=0)
        return logits

    def get_running_mu(self, x, update=True):
        mu = x.mean(dim=0).unsqueeze(1)
        if not self.opt.accurate_mu:
            return mu
        else:
            if not hasattr(self, "run_avg_mu"):
                if update:
                    self.run_avg_mu = mu
                    return mu
                else:
                    return mu.detach()
            else:
                if update:
                    self.run_avg_mu = (1.0 / 25.0) * mu + (24.0 / 25.0) * self.run_avg_mu.detach()
                    return self.run_avg_mu
                else:
                    return self.run_avg_mu.detach()

    def get_t2s_mu(self, test_loader):
        n = test_loader.length
        mu = torch.zeros(self.x_dim, 1).to(self.device)
        for i in range(n):
            inputs, labels, test_t2s = test_loader.next()
            self.set_input(None, None, None, inputs, test_t2s)
            x_t2s = self.backbone(self.t2s_mappings)
            mu += x_t2s.mean(dim=0).unsqueeze(1)
        mu /= float(n)
        return mu

    def optimize(self):
        if self.opt.baseline:
            self.linear_opt.zero_grad()
            self.backbone_opt.zero_grad()
        else:
            self.backbone_opt.zero_grad()
            self.vae_opt.zero_grad()
            self.linear_opt.zero_grad()

        # find feature maps
        fm_s = self.backbone(self.source_imgs)
        fm_t = self.backbone(self.target_imgs)

        # optimize cyclegan
        self.cyclegan.set_input(fm_s, fm_t)
        self.cyclegan.optimize_parameters()
        fm_s2t = self.cyclegan.fake_B
        fm_t2s = self.cyclegan.fake_A

        x_s2t = self.backbone.module.pool(fm_s2t)
        x_t2s = self.backbone.module.pool(fm_t2s)
        x_s = self.backbone.module.pool(fm_s)
        x_t = self.backbone.module.pool(fm_t)

        if self.opt.baseline:
            y_logits = self.linear_y(x_s2t)
            self.loss_y = self.ce(y_logits, self.source_labels)
            self.loss_y.backward()
            self.linear_opt.step()
            self.backbone_opt.step()
            _, prediction = torch.max(y_logits, 1)
            accuracy = (prediction == self.source_labels).float().mean()
        else:
            # find Z
            xp_s2t, mu_s2t, log_sigma_s2t, z_s2t = self.vae(x_s2t.detach())
            xp_t, mu_t, log_sigma_t, z_t = self.vae(x_t.detach())
            beta = self.opt.beta_vae
            self.loss_vae = self.vae.vae_loss(x_s2t.detach(), xp_s2t, mu_s2t, log_sigma_s2t, beta=beta) +\
                            self.vae.vae_loss(x_t.detach(), xp_t, mu_t, log_sigma_t, beta=beta)
            
            # self.loss_vae = self.vae.vae_loss(x_t.detach(), xp_t, mu_t, log_sigma_t)
            self.loss_vae.backward()
            self.vae_opt.step()

            # learn linear models
            if self.opt.backward_linear_loss:
                inputs = torch.cat((z_s2t.detach(), x_s2t), dim=1)
                if self.opt.use_target_estimate:
                    tildeX_inputs = torch.cat((z_t.detach(), x_t), dim=1)
                    tildeX_target = x_t2s
                else:
                    tildeX_inputs = inputs
                    tildeX_target = x_s
            else:
                inputs = torch.cat((z_s2t.detach(), x_s2t.detach()), dim=1)
                if self.opt.use_target_estimate:
                    tildeX_inputs = torch.cat((z_t.detach(), x_t.detach()), dim=1)
                    tildeX_target = x_t2s
                else:
                    tildeX_inputs = inputs
                    tildeX_target = x_s
            y_linear_preds = self.linear_y(inputs)
            x_tilde_preds = self.linear_tilde_x(tildeX_inputs)
            self.loss_linear = self.ce(y_linear_preds, self.source_labels) + self.mse(x_tilde_preds, tildeX_target.detach())
            self.loss_linear.backward(retain_graph=self.opt.backward_linear_loss)
            self.linear_opt.step()

            # compute causal effects
            b0 = self.linear_y.bias.data.unsqueeze(1)
            w1 = self.linear_y.weight.data[:, :self.z_dim]
            w2 = self.linear_y.weight.data[:, self.z_dim:]
            w3 = self.linear_tilde_x.weight.data[:, :self.z_dim]
            w3i = torch.pinverse(w3)
            w4 = self.linear_tilde_x.weight.data[:, self.z_dim:]
            b1 = self.linear_tilde_x.bias.data.unsqueeze(1)
            if self.opt.use_target_estimate:
                mu = self.get_running_mu(x_t2s)
            else:
                mu = self.get_running_mu(x_s)

            w1w3i = torch.mm(w1, w3i)
            w2w1w3w4 = w2 - torch.mm(w1w3i, w4)  # nc * x_dim
            bs = x_s2t.shape[0]
            y_logits = (b0 - torch.mm(w1w3i, b1) + torch.mm(w1w3i, mu)).expand(-1, bs).t() + torch.mm(w2w1w3w4, x_s2t.t()).t()
            self.loss_causal = self.ce(y_logits, self.source_labels)
            if not self.opt.backward_generator:
                self.backbone_opt.zero_grad()
            if self.iteration >= self.opt.pretrain_iteration:
                self.loss_causal.backward()
                self.backbone_opt.step()
            _, prediction = torch.max(y_logits, 1)
            accuracy = (prediction == self.source_labels).float().mean()

        self.iteration += 1
        self.step_schedulers()
        return accuracy

    def predict(self):
        with torch.no_grad():
            fm_s = self.backbone(self.source_imgs)
            fm_t = self.backbone(self.target_imgs)

            # Generate feature maps
            self.cyclegan.set_input(fm_s, fm_t)
            self.cyclegan.forward()
            self.cyclegan.backward_G(backward_loss=False)

            fm_t2s = self.cyclegan.fake_A
            x_t2s = self.backbone.module.pool(fm_t2s)
            x_t = self.backbone.module.pool(fm_t)
            if self.opt.baseline:
                y_logits = self.linear_y(x_t)
            else:
                b0 = self.linear_y.bias.data.unsqueeze(1)
                w1 = self.linear_y.weight.data[:, :self.z_dim]
                w2 = self.linear_y.weight.data[:, self.z_dim:]
                w3 = self.linear_tilde_x.weight.data[:, :self.z_dim]
                w3i = torch.pinverse(w3)  # (A.T.dot(A)).inv() or (A.dot(A.T)).inv() for numerical stability
                w4 = self.linear_tilde_x.weight.data[:, self.z_dim:]
                b1 = self.linear_tilde_x.bias.data.unsqueeze(1)
                if self.opt.accurate_mu:
                    mu = self.target_mu
                else:
                    mu = x_t2s.mean(dim=0).unsqueeze(1)

                w1w3i = torch.mm(w1, w3i)
                w2w1w3w4 = w2 - torch.mm(w1w3i, w4)
                bs = x_t.shape[0]
                y_logits = (b0 - torch.mm(w1w3i, b1) + torch.mm(w1w3i, mu)).expand(-1, bs).t() + torch.mm(w2w1w3w4, x_t.t()).t()
        return y_logits

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        # errors_ret = OrderedDict()
        errors_ret = self.cyclegan.get_current_losses()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def load_networks(self, load_suffix):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (load_suffix, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)
        self.cyclegan.load_networks(load_suffix)

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if name == "backbone":
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
        self.cyclegan.save_networks(epoch)

    def setup(self):
        if not self.train or self.opt.dda_continue_train:
            load_suffix = 'iter_%d' % self.opt.dda_load_iter if self.opt.dda_load_iter > 0 else self.opt.dda_epoch
            self.load_networks(load_suffix)
        self.print_networks(self.opt.debug)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        self.cyclegan.print_networks(verbose)