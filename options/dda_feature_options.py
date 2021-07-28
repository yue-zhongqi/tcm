class DDAFeatureOptions():
    # def __init__(self):
    #    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    @staticmethod
    def initialize(parser):
        # Train options
        # General
        parser.add_argument('--debug', action='store_true', help='Debug mode')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default="debug", help='Experiment name')
        parser.add_argument('--checkpoints_dir', type=str, default="/data2/xxxxx/Model/dda/dda_feature_models", help='DDA checkpoint directory')
        parser.add_argument('--dda_print_freq', type=int, default=10, help='DDA print loss frequency')
        parser.add_argument('--dda_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--dda_load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--dda_continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--snapshot_interval', type=int, default=2000, help='Test interval')

        # Data loading
        parser.add_argument('--s_name', type=str, default="A", help='source domain name')
        parser.add_argument('--t_name', type=str, default="D", help='target domain name')
        parser.add_argument('--dataset_name', type=str, default="office", help='dataset name')
        parser.add_argument('--s_dset', type=str, default="", help='source data txt file')
        parser.add_argument('--t_dset', type=str, default="", help='target data txt file')
        parser.add_argument('--batch_size', type=int, default=32, help='Train batch size')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--dda_scale_size', type=int, default=224, help='scale the loaded image to this size')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        # DDA Training
        parser.add_argument("--baseline", action="store_true", help="Use baseline for training dda (optimizing P(Y|X)).")
        parser.add_argument('--num_classes', type=int, default=31, help='Number of classes')
        parser.add_argument('--n_iterations', type=int, default=20000, help='Number of training iterations')
        parser.add_argument('--resnet_name', type=str, default="", help='ResNet name')
        parser.add_argument('--z_dim', type=int, default=100, help='Z dimension')
        parser.add_argument('--backbone_lr', type=float, default=0.0003, help='DDA backbone learning rate')
        parser.add_argument('--vae_lr', type=float, default=0.00001, help='VAE learning rate')
        parser.add_argument('--linear_lr', type=float, default=0.003, help='DDA linear learning rate')
        parser.add_argument('--linear_weight_decay', type=float, default=0.001, help='DDA linear weight decay')
        parser.add_argument('--linear_momentum', type=float, default=0.9, help='DDA linear momentum')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--backward_linear_loss', action='store_true', help='if enabled, linear loss in dda will be backward for backbone update')
        parser.add_argument('--accurate_mu', action='store_true', help='if enabled, mu will be estimated using running average during training, and accurately calculated during testing.')
        parser.add_argument('--label_smoothing', action='store_true', help='if enabled, use label smoothing instead of cross entropy.')
        parser.add_argument("--beta_vae", type=float, default=1.0, help="Beta for beta-vae.")
        parser.add_argument('--pretrain_iteration', type=int, default=0, help='Pretraining epochs for encoder and linear in dda')
        parser.add_argument('--use_target_estimate', action='store_true', help='if enabled, use t2s and x_t to train tildeX linear and mu instead of x_s and x_s2t')
        # DDA Testing
        parser.add_argument('--test_interval', type=int, default=500, help='Test interval')

        # CycleGAN
        parser.add_argument('--n_experts', type=int, default=4, help='# of experts')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=1, help='Number of discriminator conv layers')
        parser.add_argument('--n_blocks_G', type=int, default=0, help='Number of generator blocks')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--backward_generator', action='store_true', help='backward generator loss to backbone')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        parser.add_argument('--lambda_similar', type=float, default=0.0, help='the mapping must still be similar to itself')
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument("--expert_criteria", type=str, default="d", help='d | dc | dci; discriminator only; d + cycle consistency; d + cycle + identity')
        parser.add_argument("--c_criteria_iterations", type=int, default=150, help='if c in expert_criteria, after this many iterations, start using cycle consistency')
        parser.add_argument("--i_criteria_iterations", type=int, default=150, help='if i in expert_criteria, after this many iterations, start using identity')
        parser.add_argument("--expert_warmup_mode", type=str, default="none", help='none | random | classwise')
        parser.add_argument("--expert_warmup_iterations", type=int, default=100, help='warm up iterations')
        parser.add_argument("--lr_trick", type=int, default=0, help='if using lr trick')
        return parser

    @staticmethod
    def process_opt_str(opt, opt_str):
        opt_str += opt.name

        opt_str += "-%s%s2%s" % (opt.dataset_name, opt.s_name, opt.t_name)

        opt_str += "-%s" % (opt.resnet_name)

        opt_str += "-z%d" % (opt.z_dim)

        # opt_str += "-b%.4fv%.4fl%.4f" % (opt.backbone_lr, opt.vae_lr, opt.linear_lr)
            
        if opt.baseline:
            opt_str += "-baseline"
        else:
            if opt.backward_linear_loss:
                opt_str += "-bl"
            if opt.pretrain_iteration != 0:
                opt_str += "-p%d" % (opt.pretrain_iteration)
        
        if opt.accurate_mu:
            opt_str += "-mu"
        if opt.beta_vae != 1.0:
            opt_str += "-beta%.1f" % opt.beta_vae

        if opt.label_smoothing:
            opt_str += "-ls"

        if opt.use_target_estimate:
            opt_str += "-te"
        
        if opt.backward_generator:
            opt_str += "-backG"

        # CycleGAN
        opt_str += "-g%dd%d-ngf%dndf%d-%s-%s" % (opt.n_blocks_G, opt.n_layers_D, opt.ngf, opt.ndf, opt.norm, opt.gan_mode)

        opt_str += "-n%d" % opt.n_experts

        opt_str += "-lc%dli%.1fls%.1f" % (opt.lambda_A, opt.lambda_identity, opt.lambda_similar)

        opt_str += "-E%s-%s%d-lr%d" % (opt.expert_criteria, opt.expert_warmup_mode, opt.expert_warmup_iterations, opt.lr_trick)
        return opt_str