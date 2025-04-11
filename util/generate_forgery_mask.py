import os, random
from os import listdir
import cv2 as cv
import torch
from torchvision import transforms
from torchvision import utils as vutils

original_img_path = ''
obj_mask_path = ''
output_path = ''

import numpy as np
import cv2


def random_flip(image):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    if hflip:
        image = image[:, ::-1]
    if vflip:
        image = image[::-1, :]
    if rot90:
        image = image.transpose(1, 0)
    return image


def crop_obj_mask(original_img_path, obj_mask_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_lis = sorted(os.listdir(original_img_path))
    obj_mask_lis = sorted(os.listdir(obj_mask_path))
    print('img_lis:', len(img_lis))
    print('obj_mask_lis:', len(obj_mask_lis))
    for i in range(len(img_lis)):
        obj_mask_name = obj_mask_lis[i]
        img_name = img_lis[i]

        obj_mask_dir = '{}{}'.format(obj_mask_path, obj_mask_name)
        img_dir = '{}{}'.format(original_img_path, img_name)
        img = cv.imread(img_dir)
        obj_mask = cv.imread(obj_mask_dir, 0)
        _, obj_mask = cv.threshold(obj_mask, 0, 255, cv.THRESH_BINARY)

        img_h, img_w, _ = img.shape
        h, w = obj_mask.shape

        # if ((img_h > img_w) and (h < w)) or ((img_h < img_w) and (h > w)):
        #     obj_mask = cv.rotate(obj_mask, cv.ROTATE_90_CLOCKWISE)
        #     h, w = obj_mask.shape
        obj_mask = random_flip(obj_mask)

        obj_mask = cv.resize(obj_mask, (img_w, img_h))
        _, obj_mask = cv.threshold(obj_mask, 0, 255, cv.THRESH_BINARY)

        img_name = img_name.replace('.jpg', '.png')
        assert obj_mask.shape[0] == img.shape[0] and obj_mask.shape[1] == img.shape[1]
        cv.imwrite('{}{}'.format(output_path, img_name), obj_mask)
        print('saving:[{}]'.format(img_name))


if __name__ == "__main__":
    crop_obj_mask(original_img_path, obj_mask_path, output_path)
