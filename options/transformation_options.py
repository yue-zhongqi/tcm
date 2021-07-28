import argparse
import os

class TransformationOptions():
    # def __init__(self):
    #    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    @staticmethod
    def initialize(parser):
        ############################ Base options #################################
        # basic parameters
        parser.add_argument('--debug', action='store_true', help='Debug mode')
        parser.add_argument('--a_root', required=True, help='path to domain A')
        parser.add_argument('--b_root', required=True, help='path to domain B')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--exp_name', type=str, default='auto_set', help='Will be automatically set')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='/data2/xxxx/Model/dda/cyclegan', help='models are saved here')
        # model parameters
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | resnet_3blocks | resnet_2blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        #parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=224, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        #parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        #################################### Train options ######################################
        # IM options
        parser.add_argument('--n_experts', type=int, default=1, help='Number of experts.')
        parser.add_argument("--expert_criteria", type=str, default="d", help='d | dc | dci; discriminator only; d + cycle consistency; d + cycle + identity')
        parser.add_argument("--c_criteria_iterations", type=int, default=5000, help='if c in expert_criteria, after this many iterations, start using cycle consistency')
        parser.add_argument("--i_criteria_iterations", type=int, default=5000, help='if i in expert_criteria, after this many iterations, start using identity')
        parser.add_argument("--expert_warmup_mode", type=str, default="none", help='none | random | classwise')
        parser.add_argument("--expert_warmup_iterations", type=int, default=3000, help='warm up iterations')
        parser.add_argument("--lr_trick", type=int, default=1, help='if using lr trick')
        parser.add_argument("--early_stop_active_expert", action="store_true", help='if n_active expert<n_experts-1, stop')

        # Discriminator options
        parser.add_argument('--cg_resnet_name', type=str, default="ResNet50", help='ResNet name')
        parser.add_argument('--cg_num_classes', type=int, default=31, help='Number of classes')
        parser.add_argument('--cg_align_feature', action="store_true", help='Align feature for cyclegan')
        parser.add_argument('--cg_align_logits', action="store_true", help='Align logits for cyclegan')
        parser.add_argument("--cg_gvbd_weight", type=float, default=0.0, help="GVBD weight.")
        parser.add_argument("--cg_gvbg_weight", type=float, default=0.0, help="GVBG weight.")

        # cyclegan options
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--lambda_diversity', type=float, default=0.0, help='weight for diversity loss')
        # generic options
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        ##################################### Test Options ###################################
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        return parser

    @staticmethod
    def process_opt_str(opt, opt_str):
        opt_str += opt.name

        opt_str += "-ngf%dndf%d" % (opt.ngf, opt.ndf)

        if opt.netD == "basic":
            opt_str += "-netDb"
        elif opt.netD == "n_layers":
            opt_str += "-netDn%d" % (opt.n_layers_D)
        elif opt.netD == "pixel":
            opt_str += "-netDp"

        if opt.netG == "resnet_9blocks":
            opt_str += "-netG9"
        elif opt.netG == "resnet_6blocks":
            opt_str += "-netG6"
        elif opt.netG == "resnet_3blocks":
            opt_str += "-netG3"
        elif opt.netG == "resnet_2blocks":
            opt_str += "-netG2"
        elif opt.netG == "unet_256":
            opt_str += "-netG256"
        elif opt.netG == "unet_128":
            opt_str += "-netG128"

        opt_str += "-init%s%.2f" % (opt.init_type[:1], opt.init_gain)

        if opt.no_dropout:
            opt_str += "-nodrop"

        if opt.no_flip:
            opt_str += "-noflip"

        # n epochs with initial learning rate
        opt_str += "-epoch%d" % (opt.n_epochs)

        # n epochs to decay learning rate
        if opt.n_epochs_decay != 100:
            opt_str += "-decay%d" % (opt.n_epochs_decay)

        opt_str += "-batch%d" % (opt.batch_size)

        opt_str += "-lrp%s" % (opt.lr_policy[:1])

        opt_str += "-lr%.4fb%.1f" % (opt.lr, opt.beta1)

        opt_str += "-gan%s" % (opt.gan_mode[:1])

        opt_str += "-lA%.1flB%.1fli%.1f" % (opt.lambda_A, opt.lambda_B, opt.lambda_identity)

        if opt.lambda_diversity > 0:
            opt_str += "-lD%.2f" % (opt.lambda_diversity)

        # IM options
        opt_str += "-n%d" % (opt.n_experts)
        if opt.expert_criteria != "d":
            opt_str += "-%s_%d_%d" % (opt.expert_criteria, opt.c_criteria_iterations, opt.i_criteria_iterations)

        if opt.expert_warmup_mode != "none":
            opt_str += "-%s%d" % (opt.expert_warmup_mode[0], opt.expert_warmup_iterations)

        if opt.lr_trick == 0:
            opt_str += "-nolr"

        if opt.cg_align_feature or opt.cg_align_logits:
            opt_str += "-f%dl%d_g%.1fd%.1f" % (opt.cg_align_feature, opt.cg_align_logits, opt.cg_gvbg_weight, opt.cg_gvbd_weight)
            if "50" in opt.cg_resnet_name:
                opt_str += "_50"
            else:
                opt_str += "_101"
        return opt_str