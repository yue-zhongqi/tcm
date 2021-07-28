#from __future__ import print_function, division

import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
from PIL import ImageFile
import random
from dda_model.util import get_cdm_file_name, get_expert_cdm_file_name

ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', return_path=False,
                 return_cdm=False, cdm_path="", cdm_transform=None):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.return_path = return_path
        self.transform = transform
        self.target_transform = target_transform
        self.cdm_transform = cdm_transform
        self.return_cdm = return_cdm
        self.cdm_path = cdm_path
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_cdm:
            cdm_file_name = get_cdm_file_name(path)
            cdm_file_path = os.path.join(self.cdm_path, cdm_file_name)
            cdm = self.loader(cdm_file_path)
            if self.cdm_transform is not None:
                cdm = self.cdm_transform(cdm)
        if self.return_path:
            if self.return_cdm:
                return img, target, path, cdm
            else:
                return img, target, path
        else:
            if self.return_cdm:
                return img, target, cdm
            else:
                return img, target

    def __len__(self):
        return len(self.imgs)

class ExpertImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', return_path=False, cdm_path="", cdm_transform=None, n_experts=1):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.return_path = return_path
        self.transform = transform
        self.target_transform = target_transform
        self.cdm_transform = cdm_transform
        self.cdm_path = cdm_path
        self.n_experts = n_experts
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        cdms = torch.zeros(self.n_experts, img.shape[0], img.shape[1], img.shape[2]).to(img.device)
        # cdms = cdms.unsqueeze(0).expand(self.n_experts, -1, -1, -1)
        for i in range(self.n_experts):
            cdm_file_name = get_expert_cdm_file_name(path, i)
            cdm_file_path = os.path.join(self.cdm_path, cdm_file_name)
            cdm = self.loader(cdm_file_path)
            if self.cdm_transform is not None:
                cdm = self.cdm_transform(cdm)
            cdms[i] = cdm

        if self.return_path:
            return img, target, path, cdms
        else:
            return img, target, cdms

    def __len__(self):
        return len(self.imgs)


class BundledImageList(Dataset):
    def __init__(self, image_list, labels=None, ori_transform=None, cdm_transform=None, mode='RGB', return_path=False,
                 cdm_path="", bundled_transform=None, resized_crop_size=224, random_horizontal_flip=False):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.return_path = return_path
        self.ori_transform = ori_transform
        self.cdm_transform = cdm_transform
        self.bundled_transform = bundled_transform
        self.cdm_path = cdm_path
        self.resized_crop_size = resized_crop_size
        self.random_horizontal_flip = random_horizontal_flip
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.ori_transform is not None:
            img = self.ori_transform(img)

        cdm_file_name = get_cdm_file_name(path)
        cdm_file_path = os.path.join(self.cdm_path, cdm_file_name)
        cdm = self.loader(cdm_file_path)
        if self.cdm_transform is not None:
            cdm = self.cdm_transform(cdm)

        if self.random_horizontal_flip:
            if random.random() > 0.5:
                img = F.hflip(img)
                cdm = F.hflip(cdm)

        if self.resized_crop_size > 0:
            # Perform random resized crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
            size = (self.resized_crop_size, self.resized_crop_size)
            img = F.resized_crop(img, i, j, h, w, size)
            cdm = F.resized_crop(cdm, i, j, h, w, size)
        
        if self.bundled_transform is not None:
            img = self.bundled_transform(img)
            cdm = self.bundled_transform(cdm)

        if self.return_path:
            return img, target, path, cdm
        else:
            return img, target, cdm

    def __len__(self):
        return len(self.imgs)

class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)