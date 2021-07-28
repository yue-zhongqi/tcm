import os
import configparser

config_file = "configs/ic/ip_n5.txt"
configParser = configparser.RawConfigParser()
configParser.read(config_file)
cyclegan = configParser.get('dda-config', 'cyclegan')
ori_exp_name = configParser.get('dda-config', 'exp_name')
s_name = configParser.get('dda-config', 's_name')
t_name = configParser.get('dda-config', 't_name')
s_set = configParser.get('dda-config', 's_set')
t_set = configParser.get('dda-config', 't_set')
dataset_name = configParser.get('dda-config', 'dataset_name')
num_classes = configParser.get('dda-config', 'num_classes')
cdm = configParser.get('dda-config', 'cdm')
alpha = "1.0"
batch_size = "32"
epoch = "40"
gpu = "0,1,2,3"

os.system("%s --epoch %s" % (cdm, epoch))
exp_name = "alpha_%s_e%s" % (ori_exp_name, epoch)


args = ["--gpu_ids", gpu, "--s_dset", s_set, "--s_name", s_name, "--t_dset", t_set, "--t_name", t_name, "--dataset_name", dataset_name, "--name", "dda1blprlr_%s" % (exp_name), "--resnet_name", "ResNet50", "--cdt_exp_name", cyclegan, "--test_interval", "100", "--train_batch_size", batch_size, "--test_batch_size", batch_size, "--align_logits", "--gvbd_weight", "0.0", "--gvbg_weight", "0.0", "--num_classes", num_classes, "--pretrain_iteration", "300", "--backbone_lr", "0.001", "--linear_lr", "0.01", "--discriminator_lr", "0.01", "--backward_linear_loss", "--n_iterations", "3000", "--alpha", alpha]
args_str = " ".join(args)
os.system("python train_dda.py %s" % (args_str))