import os
from os import listdir
import cv2 as cv
import torch
from torchvision import transforms
from torchvision import utils as vutils
from utils import *
# from dataset import random_free_form_mask
from util.mask_generator import *

original_img_path = ''
output_path = ''


def center_crop_image(image):
    width, height, _ = image.shape
    new_height = min(width, height)
    new_width = min(width, height)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    cropped_image = image[top:bottom, left:right]
    return cropped_image


def crop_AHP(original_img_path, output_path):
    make_dir(output_path)
    for img_name in listdir(original_img_path):
        original_dir = '{}{}'.format(original_img_path, img_name)

        img = cv.imread(original_dir)

        img = center_crop_image(img)

        img = cv.resize(img, (512, 512))

        cv.imwrite('{}{}'.format(output_path, img_name), img)

        print('saving:[{}]'.format(mask_dir))


if __name__ == "__main__":
    crop_AHP(original_img_path, output_path)
