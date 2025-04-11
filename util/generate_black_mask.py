import os, random
from os import listdir
import cv2 as cv
import torch
from torchvision import transforms
from torchvision import utils as vutils

original_img_path = ''
output_path = ''

import numpy as np
import cv2


def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def crop_obj_mask(original_img_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_lis = sorted(os.listdir(original_img_path))
    print('img_lis:', len(img_lis))
    for i in range(len(img_lis)):
        img_name = img_lis[i]
        img_h, img_w = 512, 512
        img_dir = '{}{}'.format(original_img_path, img_name)
        if i == 0:
            img = cv.imread(img_dir)
            img_h, img_w, _ = img.shape

        obj_mask = np.zeros((img_h, img_w, 1), np.uint8)
        _, obj_mask = cv.threshold(obj_mask, 0, 255, cv.THRESH_BINARY)
        assert obj_mask.shape[0] == img_h and obj_mask.shape[1] == img_w
        img_name = img_name.replace('.jpg', '.png')
        cv.imwrite('{}{}'.format(output_path, img_name), obj_mask)
        print('saving:[{}]'.format(img_name))


if __name__ == "__main__":
    crop_obj_mask(original_img_path, output_path)
