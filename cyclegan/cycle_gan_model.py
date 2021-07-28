import torch
import itertools
from collections import OrderedDict
from cyclegan.util import ImagePool
from .base_model import BaseModel
from cyclegan.cycle_gan_discriminator import CycleGANDiscriminator
from . import networks
import torch.nn as nn
import numpy as np


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'diversity']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.n_experts)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.n_experts)

        if self.opt.cg_align_feature or self.opt.cg_align_logits:
            self.discriminate_feature = True
            self.discriminator = CycleGANDiscriminator(self.opt)
        else:
            self.discriminate_feature = False

        self.n_experts = opt.n_experts
        self.panel_tracker = np.ones(self.n_experts)    # Track total number of expert selection; Add 1 for numerical stability
        self.epoch_panel_tracker = np.zeros(self.n_experts)  # Track epoch-wise expert selection
        self.iteration = 0
        self.cdm_mode = False   # Switch of CDM mode; if on, expert will not use random or classwise
        if self.opt.expert_warmup_mode == "classwise":
            self.classwise_assignment = {}
            self.classwise_expert_idx = 0

        if self.isTrain:  # define discriminators
        # if True:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
        # if True:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss(reduction='none')
            self.criterionIdt = torch.nn.L1Loss(reduction='none')
            self.criterionDiversity = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def end_epoch(self):
        self.epoch_panel_tracker = np.zeros(self.n_experts)

    def get_expert_selection_results(self):
        results = OrderedDict()
        for i in range(self.n_experts):
            name = "expert_%d" % i
            value = self.epoch_panel_tracker[i]
            results[name] = value
        return results

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.current_batch_size = self.real_A.shape[0]
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.a_paths = input['A_paths']
        self.b_paths = input['B_paths']
        self.update_input_classname()

    def update_input_classname(self):
        self.input_classnames = []
        for i in range(self.current_batch_size):
            self.input_classnames.append(self.a_paths[i].split("/")[-2])
        for i in range(self.current_batch_size):
            self.input_classnames.append(self.b_paths[i].split("/")[-2])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_all = self.netG_A(self.real_A.unsqueeze(0).expand(self.n_experts, -1, -1, -1, -1))  # G_A(A)
        self.rec_A_all = self.netG_B(self.fake_B_all)   # G_B(G_A(A))
        self.fake_A_all = self.netG_B(self.real_B.unsqueeze(0).expand(self.n_experts, -1, -1, -1, -1))  # G_B(B)
        self.rec_B_all = self.netG_A(self.fake_A_all)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_real_basic(self, netD, real):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * 0.5
        loss_D_real = loss_D_real.mean()
        loss_D_real.backward()
        return loss_D_real

    def backward_D_fake_basic(self, netD, fake):
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * 0.5 / float(self.n_experts)
        loss_D_fake = loss_D_fake.mean()
        loss_D_fake.backward()
        return loss_D_fake

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B)
        #self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        loss_D_A = self.backward_D_real_basic(self.netD_A, self.real_B)
        for i in range(0, self.n_experts):
            fake_B = self.fake_B_pool.query(self.fake_B_all[i])
            loss_D_A += self.backward_D_fake_basic(self.netD_A, fake_B)
        self.loss_D_A = loss_D_A

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        #fake_A = self.fake_A_pool.query(self.fake_A)
        #self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        loss_D_B = self.backward_D_real_basic(self.netD_B, self.real_A)
        for i in range(0, self.n_experts):
            fake_A = self.fake_A_pool.query(self.fake_A_all[i])
            loss_D_B += self.backward_D_fake_basic(self.netD_B, fake_A)
        self.loss_D_B = loss_D_B

    def backward_G(self, backward_loss=True):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        current_batch_size = self.real_A.shape[0]
        # Identity loss
        losses = torch.zeros(self.n_experts, current_batch_size * 2).to(self.device)   # n * batch_size
        self.loss_G_A = []
        self.loss_G_B = []
        self.loss_cycle_A = []
        self.loss_cycle_B = []
        self.loss_idt_A = []
        self.loss_idt_B = []
        for i in range(0, self.n_experts):
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                self.idt_A = self.netG_A.get_expert(i)(self.real_B)
                self.loss_idt_A.append(self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt)
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B.get_expert(i)(self.real_A)
                self.loss_idt_B.append(self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt)
            else:
                self.loss_idt_A.append(torch.zeros(current_batch_size, 1, 1, 1).to(self.device))
                self.loss_idt_B.append(torch.zeros(current_batch_size, 1, 1, 1).to(self.device))

            # GAN loss D_A(G_A(A))
            self.loss_G_A.append(self.criterionGAN(self.netD_A(self.fake_B_all[i]), True))
            # GAN loss D_B(G_B(B))
            self.loss_G_B.append(self.criterionGAN(self.netD_B(self.fake_A_all[i]), True))
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A.append(self.criterionCycle(self.rec_A_all[i], self.real_A) * lambda_A)
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B.append(self.criterionCycle(self.rec_B_all[i], self.real_B) * lambda_B)
            losses[i] = self.get_expert_loss(i)
        expert_idx = self.get_expert_results(losses)    # 2batch_size
        self.loss_G = 0
        for i in range(0, current_batch_size):
            if self.opt.lr_trick:
                a_expert_loss_scale = float(self.panel_tracker.sum() / float(self.panel_tracker[expert_idx[i]]) / float(self.n_experts))
                self.loss_G += (self.loss_G_A[expert_idx[i]][i].mean() + self.loss_cycle_A[expert_idx[i]][i].mean() + self.loss_idt_A[expert_idx[i]][i].mean()) / a_expert_loss_scale
                b_expert_loss_scale = float(self.panel_tracker.sum() / float(self.panel_tracker[expert_idx[i+current_batch_size]]) / float(self.n_experts))
                self.loss_G += (self.loss_G_B[expert_idx[i+current_batch_size]][i].mean() + self.loss_cycle_B[expert_idx[i+current_batch_size]][i].mean() + self.loss_idt_B[expert_idx[i+current_batch_size]][i].mean()) / b_expert_loss_scale
            else:
                self.loss_G += (self.loss_G_A[expert_idx[i]][i].mean() + self.loss_G_B[expert_idx[i+current_batch_size]][i].mean() + self.loss_cycle_A[expert_idx[i]][i].mean() + self.loss_cycle_B[expert_idx[i+current_batch_size]][i].mean() + self.loss_idt_A[expert_idx[i]][i].mean() + self.loss_idt_B[expert_idx[i+current_batch_size]][i].mean())
        self.loss_G /= float(current_batch_size)
        
        # select fake and reconstruct images
        fake_B = torch.zeros(self.fake_B_all[0].shape).to(self.device)
        rec_A = torch.zeros(self.rec_A_all[0].shape).to(self.device)
        fake_A = torch.zeros(self.fake_A_all[0].shape).to(self.device)
        rec_B = torch.zeros(self.rec_B_all[0].shape).to(self.device)
        for i in range(0, self.real_A.shape[0]):
            fake_B[i] = self.fake_B_all[expert_idx[i]][i]
            rec_A[i] = self.rec_A_all[expert_idx[i]][i]
            fake_A[i] = self.fake_A_all[expert_idx[i+current_batch_size]][i]
            rec_B[i] = self.rec_B_all[expert_idx[i+current_batch_size]][i]

        # convert loss to mean
        loss_diversity = 0
        for i in range(0, self.n_experts):
            self.panel_tracker[i] += (expert_idx == i).sum()
            self.epoch_panel_tracker[i] += (expert_idx == i).sum()
            self.loss_G_A[i] = self.loss_G_A[i].mean()
            self.loss_G_B[i] = self.loss_G_B[i].mean()
            self.loss_cycle_A[i] = self.loss_cycle_A[i].mean()
            self.loss_cycle_B[i] = self.loss_cycle_B[i].mean()
            self.loss_idt_A[i] = self.loss_idt_A[i].mean()
            self.loss_idt_B[i] = self.loss_idt_B[i].mean()
            loss_diversity += self.criterionDiversity(fake_B, self.fake_B_all[i]) * lambda_A\
                              + self.criterionDiversity(fake_A, self.fake_A_all[i]) * lambda_B
        loss_diversity /= -float(self.n_experts)
        self.loss_diversity = loss_diversity
        self.loss_G += loss_diversity * self.opt.lambda_diversity
        if backward_loss:
            self.loss_G.backward(retain_graph=self.discriminate_feature)

        self.fake_B = fake_B
        self.rec_A = rec_A
        self.fake_A = fake_A
        self.rec_B = rec_B
        self.expert_idx = expert_idx

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        if self.discriminate_feature:
            self.discriminator.set_input(self.real_A, None, self.fake_B, self.real_B, self.fake_A)
            self.discriminator.optimize()
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.iteration += 1

    def get_current_losses(self):
        errors_ret = super(CycleGANModel, self).get_current_losses()
        if self.discriminate_feature:
            errors_ret["loss_transfer"] = self.discriminator.loss_transfer
            errors_ret["loss_gvbg"] = self.discriminator.loss_gvbg
            errors_ret["loss_gvbd"] = self.discriminator.loss_gvbd
        return errors_ret

    def get_expert_loss(self, idx):
        loss = torch.cat((self.loss_G_A[idx].mean([1,2,3]), self.loss_G_B[idx].mean([1,2,3])), dim=0)   # 2batchsize
        if 'c' in self.opt.expert_criteria and self.iteration > self.opt.c_criteria_iterations:
            loss += torch.cat((self.loss_cycle_A[idx].mean([1,2,3]), self.loss_cycle_B[idx].mean([1,2,3])), dim=0)
            #print("using c")
        if 'i' in self.opt.expert_criteria and self.iteration > self.opt.i_criteria_iterations:
            loss += torch.cat((self.loss_idt_A[idx].mean([1,2,3]), self.loss_idt_B[idx].mean([1,2,3])), dim=0)
            #print("using i")
        return loss

    def get_expert_results(self, losses):
        _, expert_idx = losses.min(dim=0)  # batch_size
        current_batch_size = expert_idx.shape[0]
        if self.opt.expert_warmup_mode == "none" or self.iteration > self.opt.expert_warmup_iterations or self.cdm_mode:
            #print("expert using none")
            return expert_idx
        else:
            if self.opt.expert_warmup_mode == "random":
                #print("expert using random")
                return np.random.randint(0, self.n_experts, size=current_batch_size)
            if self.opt.expert_warmup_mode == "classwise":
                expert_idx = np.zeros(current_batch_size, dtype="uint8")
                for i in range(current_batch_size):
                    classname = self.input_classnames[i]
                    if classname in self.classwise_assignment:
                        expert_idx[i] = self.classwise_assignment[classname]
                    else:
                        self.classwise_assignment[classname] = self.classwise_expert_idx
                        self.classwise_expert_idx = (self.classwise_expert_idx + 1) % self.n_experts
                        expert_idx[i] = self.classwise_assignment[classname]
                return expert_idx
