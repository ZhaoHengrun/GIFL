import os
from os import listdir
import cv2 as cv
import torch
from torchvision import transforms
from torchvision import utils as vutils
import cv2
import torch.utils.data as data
import random
import numpy as np
import copy
from PIL import Image, ImageFile, ImageOps


def deepfill_mask(h, w, mv=10, ma=4.0, ml=80, mbw=20):  # training: mv:10, ma:4.0, ml:80, mbw:20
    """Generate a random free form mask with configuration. default: img_shape:[256,256], mv:5, ma:4.0, ml:40, mbw:10
    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
        :param w:
        :param h:
    """
    mask = np.zeros((h, w))
    num_v = 12 + np.random.randint(mv)  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for j in range(1 + np.random.randint(mv)):
            angle = 0.01 + np.random.randint(ma)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(ml)
            brush_w = 10 + np.random.randint(mbw)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y
    mask = mask.reshape(mask.shape + (1,)).astype(np.float32)
    mask = transforms.ToTensor()(mask.copy())
    return mask


def random_regular_mask(h, w):
    mask = torch.ones(h, w).unsqueeze(0)
    s = [1, h, w]
    N_mask = random.randint(1, 5)
    lim_x = s[1] - s[1] / (N_mask + 1)
    lim_y = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(lim_x))
        y = random.randint(0, int(lim_y))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), min(int(s[1] - x), int(s[1] / 2)))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), min(int(s[2] - y), int(s[2] / 2)))
        mask[:, int(x): int(range_x), int(y): int(range_y)] = 0
    return mask


def center_mask(h, w):
    mask = torch.ones(h, w).unsqueeze(0)
    s = [1, h, w]
    mask[:, int(s[1] / 4):int(s[1] * 3 / 4), int(s[2] / 4):int(s[2] * 3 / 4)] = 0
    return mask


def random_irregular_mask(h, w):
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones(h, w).unsqueeze(0)
    s = [1, h, w]
    img = np.zeros((s[1], s[2], 1), np.uint8)

    max_width = int(min(s[1] / 10, s[2] / 10))
    N_mask = random.randint(16, 64)
    for _ in range(N_mask):
        model = random.random()
        if model < 0.2:  # Draw random lines
            x1, x2 = random.randint(1, s[1]), random.randint(1, s[1])
            y1, y2 = random.randint(1, s[2]), random.randint(1, s[2])
            thickness = random.randint(2, max_width)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)
        elif (model > 0.2 and model < 0.5):  # Draw random circles
            x1, y1 = random.randint(1, s[1]), random.randint(1, s[2])
            radius = random.randint(2, max_width)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)
        else:  # draw random ellipses
            x1, y1 = random.randint(1, s[1]), random.randint(1, s[2])
            s1, s2 = random.randint(1, s[1]), random.randint(1, s[2])
            a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
            thickness = random.randint(2, max_width)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    img = img.reshape(s[2], s[1])
    img = Image.fromarray(img * 255)

    img_mask = transform(img)
    for j in range(s[0]):
        mask[j, :, :] = img_mask

    return mask


def gen_mask(h, w):
    """load the mask for image completion task"""
    mask_type = [0, 1, 2, 3]
    mask_type_index = random.randint(0, len(mask_type) - 1)
    mask_type = mask_type[mask_type_index]

    if mask_type == 0:  # center mask
        if random.random() > 0.3:
            return random_regular_mask(h, w)  # random regular mask
        return center_mask(h, w)
    elif mask_type == 1:  # random regular mask
        return random_regular_mask(h, w)
    elif mask_type == 2:  # random irregular mask
        return random_irregular_mask(h, w)
    elif mask_type == 3:  # deepfill_mask
        return deepfill_mask(h, w)
