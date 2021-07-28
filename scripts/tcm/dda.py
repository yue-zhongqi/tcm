import os
import configparser


######### MODIFY HERE!!!#######
config_file_list = ["configs/oh/a2c.txt"]
batch_size = "32"
epochs = ["40", "80"]
###############################
for config_file in config_file_list:
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

    for epoch in epochs:
        os.system("%s --epoch %s" % (cdm, epoch))
        exp_name = "%s_e%s" % (ori_exp_name, epoch)

        args = ["--gpu_ids", "0,1,2,3", "--s_dset", s_set, "--s_name", s_name, "--t_dset", t_set, "--t_name", t_name, "--dataset_name", dataset_name, "--name", "dda4llt2s_%s" % (exp_name), "--num_classes", "31", "--resnet_name", "ResNet50", "--cdt_exp_name", cyclegan, "--test_interval", "100", "--train_batch_size", batch_size, "--test_batch_size", batch_size, "--align_logits", "--gvbd_weight", "0.0", "--gvbg_weight", "0.0", "--num_classes", num_classes, "--use_linear_logits", "--align_t2s", "--n_iterations", "15000"]
        args_str = " ".join(args)
        os.system("python train_dda.py %s" % (args_str))