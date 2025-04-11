import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
from os import listdir
import random
from random import randint

import cv2 as cv
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import time
from utils import *
from torchvision import utils as vutils
import albumentations as A
from util.mask_generator import gen_mask


class ValidDataset(data.Dataset):
    def __init__(self, img_path, mask_path, resize, test_length=-1):
        self.img_path = img_path
        self.mask_path = mask_path
        self.resize = resize

        if test_length != -1:
            self.img_lis = sorted(os.listdir(self.img_path))[:test_length]
            self.mask_lis = sorted(os.listdir(self.mask_path))[:test_length]
        else:
            self.img_lis = sorted(os.listdir(self.img_path))
            self.mask_lis = sorted(os.listdir(self.mask_path))
        if len(self.img_lis) != len(self.mask_lis):
            print('Warning: len(self.img_lis) != len(self.mask_lis)', )

    def __getitem__(self, index):
        img_name = self.img_lis[index]
        mask_name = self.mask_lis[index]
        full_img_path = '{}{}'.format(self.img_path, img_name)
        full_mask_path = '{}{}'.format(self.mask_path, mask_name)
        img = cv.imread(full_img_path)
        mask = cv.imread(full_mask_path)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)

        h, w = img.shape[0], img.shape[1]
        # img = cv.resize(img, (self.resize, self.resize))
        # mask = cv.resize(mask, (self.resize, self.resize))

        img = transforms.ToTensor()(img.copy())
        mask = transforms.ToTensor()(mask.copy())
        img = F.interpolate(img.unsqueeze(0), size=(self.resize, self.resize)).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.resize, self.resize)).squeeze(0)
        return img, mask, h, w, img_name

    def __len__(self):
        return len(self.img_lis)


class TrainDataset(data.Dataset):
    def __init__(self, mask_path_lis, img_path_lis, resize):
        self.mask_path_lis = mask_path_lis
        self.img_path_lis = img_path_lis
        self.resize = resize

        self.full_img_name_lis = []
        for img_path in self.img_path_lis:
            img_lis = sorted(os.listdir(img_path))
            for img_name in img_lis:
                full_img_name = '{}{}'.format(img_path, img_name)
                self.full_img_name_lis.append(full_img_name)
            print('dataset: {}, {} images'.format(img_path, len(img_lis)))

        self.full_mask_name_lis = []
        for mask_path in self.mask_path_lis:
            mask_lis = sorted(os.listdir(mask_path))
            for mask_name in mask_lis:
                full_mask_name = '{}{}'.format(mask_path, mask_name)
                self.full_mask_name_lis.append(full_mask_name)

        print('total {} images'.format(len(self.full_img_name_lis)))
        if len(self.full_img_name_lis) != len(self.full_mask_name_lis):
            print('===========Warning: len(self.img_lis) != len(self.mask_lis)', )
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),

            A.Blur(p=0.1),
            A.GaussianBlur(p=0.1),
            A.MedianBlur(p=0.1),
            A.MotionBlur(p=0.1),

            A.CLAHE(p=0.1),
            A.Sharpen(p=0.1),
            A.Emboss(p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.HueSaturationValue(p=0.1),
            A.Downscale(p=0.1),
            A.GaussNoise(p=0.1),
            A.ISONoise(p=0.1),
            A.RandomGamma(p=0.1),
            A.RandomToneCurve(p=0.1),
            A.Sharpen(p=0.1),
            A.GaussNoise(p=0.1),

            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.1),
            A.GridDistortion(p=0.1),
            A.OpticalDistortion(p=0.1),
            A.PiecewiseAffine(p=0.1),
        ])

    def __getitem__(self, index):
        img = cv.imread(self.full_img_name_lis[index])
        mask = cv.imread(self.full_mask_name_lis[index])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
        _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)

        # if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
        #     mask = cv2.resize(mask, dsize=(img.shape[1], img.shape[0]))
        h, w = img.shape[0], img.shape[1]
        # img = cv.resize(img, (self.resize, self.resize))
        # mask = cv.resize(mask, (self.resize, self.resize))

        twin_trans = self.transform(image=img, mask=mask)
        img, mask = twin_trans['image'], twin_trans['mask']
        img = transforms.ToTensor()(img.copy())
        mask = transforms.ToTensor()(mask.copy())
        img = F.interpolate(img.unsqueeze(0), size=(self.resize, self.resize)).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.resize, self.resize)).squeeze(0)
        return img, mask, h, w

    def __len__(self):
        return len(self.full_img_name_lis)
