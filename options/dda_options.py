class DDAOptions():
    # def __init__(self):
    #    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    @staticmethod
    def initialize(parser):
        # Train options
        # General
        parser.add_argument('--debug', action='store_true', help='Debug mode')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default="debug", help='Experiment name')
        parser.add_argument('--dda_checkpoints_dir', type=str, default="/data2/xxxxx/Model/dda/dda_models", help='DDA checkpoint directory')
        parser.add_argument('--dda_print_freq', type=int, default=10, help='DDA print loss frequency')
        parser.add_argument('--dda_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--dda_load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--dda_continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--snapshot_interval', type=int, default=2000, help='Test interval')

        # Data loading
        parser.add_argument("--cdt_exp_name", type=str, default="", help="Cross domain transformation experiment name")
        parser.add_argument('--s_name', type=str, default="A", help='source domain name')
        parser.add_argument('--t_name', type=str, default="D", help='target domain name')
        parser.add_argument('--dataset_name', type=str, default="office", help='dataset name')
        parser.add_argument('--cdm_path', type=str, default="/data2/xxxxx/Model/dda/cdm", help='location of cross-domain mappings')
        parser.add_argument('--s_dset', type=str, default="", help='source data txt file')
        parser.add_argument('--t_dset', type=str, default="", help='target data txt file')
        parser.add_argument('--train_batch_size', type=int, default=36, help='Train batch size')
        parser.add_argument('--test_batch_size', type=int, default=36, help='Test batch size')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--dda_scale_size', type=int, default=224, help='scale the loaded image to this size')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        # DDA Training
        parser.add_argument("--baseline", action="store_true", help="Use baseline for training dda (optimizing P(Y|X)).")
        parser.add_argument('--update_backbone', type=int, default=1, help='when using baseline, whether to update backbone')
        parser.add_argument('--no_mapping', type=int, default=0, help='when using baseline, use source instead of s2t mapping')
        parser.add_argument('--backbone_train_mode', type=int, default=0, help='Always keep backbone at train mode')
        parser.add_argument('--num_classes', type=int, default=31, help='Number of classes')
        parser.add_argument('--n_iterations', type=int, default=10000, help='Number of training iterations')
        parser.add_argument('--resnet_name', type=str, default="", help='ResNet name')
        parser.add_argument("--use_maxpool", action="store_true", help="Use maxpooling in resnet last layer")
        parser.add_argument("--freeze_layer1", action="store_true", help="Freeze layer1 in backbone")
        parser.add_argument('--z_dim', type=int, default=100, help='Z dimension')
        parser.add_argument('--backbone_lr', type=float, default=0.0003, help='DDA backbone learning rate')
        parser.add_argument('--vae_lr', type=float, default=0.00001, help='VAE learning rate')
        parser.add_argument('--linear_lr', type=float, default=0.003, help='DDA linear learning rate')
        parser.add_argument('--linear_weight_decay', type=float, default=0.001, help='DDA linear weight decay')
        parser.add_argument('--linear_momentum', type=float, default=0.9, help='DDA linear momentum')
        parser.add_argument('--dda_init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--dda_init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument("--align_feature", action="store_true", help="Add an alignment loss such that s2t resembles t features.")
        parser.add_argument("--align_logits", action="store_true", help="Add an alignment loss on logits such that s2t resembles t features.")
        parser.add_argument("--align_t2s", action="store_true", help="Additionally align t2s and s")
        parser.add_argument('--discriminator_hidden_dim', type=int, default=1024, help='Hidden dim for discriminator network')
        parser.add_argument('--discriminator_lr', type=float, default=0.003, help='Discriminator network lr')
        parser.add_argument("--gvbd_weight", type=float, default=0.0, help="GVBD weight.")
        parser.add_argument("--gvbg_weight", type=float, default=0.0, help="GVBG weight.")
        parser.add_argument("--alignment_weight", type=float, default=1.0, help="If use align feature, the weight for alignment loss.")
        parser.add_argument('--backward_linear_loss', action='store_true', help='if enabled, linear loss in dda will be backward for backbone update')
        parser.add_argument('--accurate_mu', action='store_true', help='if enabled, mu will be estimated using running average during training, and accurately calculated during testing.')
        parser.add_argument('--no_entropy_weight', action='store_true', help='if enabled, there will be no entropy reweighing when calculating loss.')
        parser.add_argument('--label_smoothing', action='store_true', help='if enabled, use label smoothing instead of cross entropy.')
        parser.add_argument("--beta_vae", type=float, default=1.0, help="Beta for beta-vae.")
        parser.add_argument('--pretrain_iteration', type=int, default=0, help='Pretraining epochs for encoder and linear in dda')
        parser.add_argument('--bundle_transform', action='store_true', help='if enabled, original image and cdm will be transformed together')
        parser.add_argument('--bundle_resized_crop', action='store_true', help='if enabled, use random resized crop on original image and cdm')
        parser.add_argument('--use_target_estimate', action='store_true', help='if enabled, use t2s and x_t to train tildeX linear and mu instead of x_s and x_s2t')
        parser.add_argument('--use_linear_logits', action='store_true', help='use linear layer to produce logits for alignment')
        parser.add_argument('--use_dda2', action='store_true', help='Use DDA model2')
        parser.add_argument('--use_dropout', action='store_true', help='Dropout for tilde x')
        parser.add_argument('--all_experts', action="store_true", help="Use all experts")
        parser.add_argument('--n_experts', type=int, default=0, help='Pretraining epochs for encoder and linear in dda')
        # DDA Testing
        parser.add_argument('--test_interval', type=int, default=500, help='Test interval')
        return parser

    @staticmethod
    def process_opt_str(opt, opt_str):
        opt_str += opt.name

        opt_str += "-%s%s2%s" % (opt.dataset_name, opt.s_name, opt.t_name)

        opt_str += "-%s" % (opt.resnet_name)

        opt_str += "-z%d" % (opt.z_dim)

        # opt_str += "-b%.4fv%.4fl%.4f" % (opt.backbone_lr, opt.vae_lr, opt.linear_lr)

        if opt.align_feature:
            opt_str += "-af"
        if opt.align_logits:
            opt_str += "=al"
        if opt.align_feature or opt.align_logits:
            opt_str += "-d%d" % (opt.discriminator_hidden_dim)
            # opt_str += "-dlr%.4f" % (opt.discriminator_lr)
            opt_str += "-w%.1f" % (opt.alignment_weight)
            opt_str += "-gvbd%.1f" % (opt.gvbd_weight)
            opt_str += "-gvbg%.1f" % (opt.gvbg_weight)
            
        if opt.baseline:
            opt_str += "-baseline%d%d%d" % (opt.update_backbone, opt.no_mapping, opt.backbone_train_mode)
        else:
            if opt.backward_linear_loss:
                opt_str += "-bl"
            if opt.pretrain_iteration != 0:
                opt_str += "-p%d" % (opt.pretrain_iteration)
        
        if opt.accurate_mu:
            opt_str += "-mu"
        if opt.beta_vae != 1.0:
            opt_str += "-beta%.1f" % opt.beta_vae

        if opt.no_entropy_weight:
            opt_str += "-ne"

        if opt.label_smoothing:
            opt_str += "-ls"

        if opt.bundle_transform:
            if not opt.bundle_resized_crop:
                opt_str += "-bt"
            else:
                opt_str += "-btr"

        if opt.use_target_estimate:
            opt_str += "-te"
            
        if opt.use_linear_logits:
            opt_str += "-ll"
        
        if opt.use_maxpool:
            opt_str += "-max"

        if opt.freeze_layer1:
            opt_str += "f1"

        if opt.use_dda2:
            opt_str += "-2"

        if opt.use_dropout:
            opt_str += "-dr"

        if opt.all_experts:
            opt_str += "-ae%d" % (opt.n_experts)
        return opt_str