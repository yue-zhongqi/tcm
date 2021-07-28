import os

nexperts = [3]

for n_expert in nexperts:
    name = "ic_cp_n%dG2" % (n_expert)
    os.system("python train_transformation.py --gpu_ids 4,5,6,7 --a_root /data2/yuezhongqi/Dataset/da/imageclef_m/c --b_root /data2/yuezhongqi/Dataset/da/imageclef_m/p --checkpoints_dir /data2/yuezhongqi/Model/dda/cyclegan --batch_size 8 --n_experts %d --name %s --lr 0.0002 --netG resnet_2blocks --display_freq 2000 --n_epochs 100 --n_epochs_decay 100 --expert_criteria dc --expert_warmup_mode random --expert_warmup_iterations 999999 --lr_trick 0 --lambda_diversity 0.01" % (n_expert, name))