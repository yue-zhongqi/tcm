import os

os.system("python train_transformation.py --gpu_ids 0,1,2,3 --a_root /data2/xxxx/Dataset/da/officehome/OfficeHomeDataset_10072016/Art --b_root /data2/xxxxx/Dataset/da/officehome/OfficeHomeDataset_10072016/Clipart --batch_size 4 --name ohac --lr 0.0002 --netG resnet_2blocks")