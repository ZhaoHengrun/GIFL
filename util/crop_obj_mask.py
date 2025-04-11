import os
from os import listdir
import cv2 as cv
import torch
from torchvision import transforms
from torchvision import utils as vutils
from utils import *
from dataset import random_free_form_mask
from util.mask_generator import *

original_img_path = ''
obj_mask_path = ''
output_path = ''


def crop_obj_mask(original_img_path, obj_mask_path, output_path):
    make_dir(output_path)
    img_lis = sorted(os.listdir(original_img_path))
    obj_mask_lis = sorted(os.listdir(obj_mask_path))
    print('img_lis:', len(img_lis))
    print('obj_mask_lis:', len(obj_mask_lis))
    for i in range(len(img_lis)):
        obj_mask_name = obj_mask_lis[i]
        img_name = img_lis[i]

        obj_mask_dir = '{}{}'.format(obj_mask_path, obj_mask_name)

        obj_mask = cv.imread(obj_mask_dir, 0)
        _, obj_mask = cv.threshold(obj_mask, 0, 255, cv.THRESH_BINARY)

        h, w = obj_mask.shape
        top = h
        buttom = 0
        left = w
        right = 0
        for i in range(h):
            for j in range(w):
                if obj_mask[i, j] == 0:
                    # 按照像素值查找需要剪切图片的起始位置，现该位置像素值为[0,0,0]
                    continue
                else:
                    # 获得剪切起始位置
                    if j <= left:
                        left = j
                    if j >= right:
                        right = j
                    if i >= buttom:
                        buttom = i
                    if i <= top:
                        top = i
        new_h = buttom - top
        new_w = right - left
        if new_h > new_w:
            padding_w = new_h - new_w
            half_padding_w = int(padding_w / 2)
            if left - half_padding_w >= 0:
                left = left - half_padding_w
            else:
                left = 0
            if left + new_h < w:
                right = left + new_h
            else:
                right = w
        if new_w > new_h:
            padding_h = new_w - new_h
            half_padding_h = int(padding_h / 2)
            if top - half_padding_h >= 0:
                top = top - half_padding_h
            else:
                top = 0
            if top + new_w < h:
                buttom = top + new_w
            else:
                buttom = h
        padded_h = buttom - top
        padded_w = right - left

        obj_mask = obj_mask[top:buttom, left:right]

        if padded_h > padded_w:
            obj_mask = cv.copyMakeBorder(obj_mask, 0, 0, 0, int(padded_h - padded_w), cv2.BORDER_REFLECT)
        if padded_w > padded_h:
            obj_mask = cv.copyMakeBorder(obj_mask, 0, int(padded_w - padded_h), 0, 0, cv2.BORDER_REFLECT)

        obj_mask = cv.resize(obj_mask, (256, 256))

        cv.imwrite('{}{}'.format(output_path, img_name), obj_mask)
        print('saving:[{}]'.format(img_name))


if __name__ == "__main__":
    crop_obj_mask(original_img_path, mask_path, obj_mask_path, output_path)
