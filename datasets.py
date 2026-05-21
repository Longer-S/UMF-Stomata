#! /usr/bin/python3

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
from PIL import Image

from torchvision import transforms
import random
import numbers
from os import listdir, mkdir
from os.path import isfile, join, isdir
import cv2

class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""

    def __init__(self, root_dir, img_list, transform_fnc=None, img_num=2):
        self.root_dir = root_dir
        self.transform_fnc = transform_fnc

        self.img_num = img_num

        self.max_n_stack = img_num

        self.imglist_all = img_list


    def __len__(self):
        return int(len(self.imglist_all)/self.max_n_stack)

    def __getitem__(self, idx):

        img_num = min(self.max_n_stack, self.img_num)
        ind = idx * img_num

        num_list = list(range(self.max_n_stack))


        mats_input = []

        for i in range(self.max_n_stack):
            im = Image.open(self.root_dir + self.imglist_all[ind + num_list[i]])#.convert('L')
            mat_all = np.array(im, dtype=np.float32)/255  # shape (H, W)
            mat_all = torch.from_numpy(mat_all)
            mats_input.append(mat_all)

        mats_input = np.stack(mats_input)

        sample = {'input': mats_input}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)

        return sample['input'], os.path.splitext(os.path.basename(self.imglist_all[ind + num_list[i]]))[0].split('_')[0]

class ToTensor(object):
    def __call__(self, sample):
        mats_input = sample['input']

        mats_input = mats_input.transpose((0, 3, 1, 2))
        # mats_output = mats_output.transpose((2, 0, 1))

        return {'input': mats_input}


class RandomCrop(object):
    """ Randomly crop images
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        inputs = sample['input']
        n, h, w, _ = inputs.shape
        th, tw = self.size
        if w < tw: tw=w
        if h < th: th=h

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = inputs[:, y1: y1 + th,x1: x1 + tw]
        return {'input':inputs}


class RandomFilp(object):
    """ Randomly crop images
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        inputs = sample['input']

        # hori filp
        if np.random.binomial(1, self.ratio):
            inputs = inputs[:,:, ::-1]


        # vert flip
        if np.random.binomial(1, self.ratio):
            inputs = inputs[:, ::-1]


        return {'input': np.ascontiguousarray(inputs)}



def FusionLoader(train_dir,val_dir, n_stack):

    img_train_list = [
        f for f in listdir(train_dir)
        if isfile(join(train_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ]
        # img_train_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "All.tif" and int(f[:6]) < 400]
    img_train_list.sort()

    img_val_list = [
        f for f in listdir(val_dir)
        if isfile(join(val_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ]
    img_val_list.sort()

    train_transform = transforms.Compose([
                        # RandomCrop(256),
                        # RandomFilp(0.5),
                        ToTensor()])
    dataset_train = ImageDataset(root_dir=train_dir, img_list=img_train_list,transform_fnc=train_transform, img_num=n_stack)

    val_transform = transforms.Compose([ToTensor()])
    dataset_valid = ImageDataset(root_dir=val_dir, img_list=img_val_list,transform_fnc=val_transform, img_num=n_stack)


    return dataset_train, dataset_valid