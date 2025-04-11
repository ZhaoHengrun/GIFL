import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
import cv2 as cv
from torchvision import utils as vutils
from PIL import Image
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import shutil
from glob import glob


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


def pixel_unshuffle(input, downscale_factor):
    """
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    """
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        """
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        """

        return pixel_unshuffle(input, self.downscale_factor)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# x = PixelUnshuffle.pixel_unshuffle(y, 2)

def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # input_tensor = input_tensor[:, [2, 1, 0], :, :]
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # input_tensor = cv.cvtColor(input_tensor, cv.COLOR_RGB2BGR)
    cv.imwrite(filename, input_tensor)


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor[:, [2, 1, 0], :, :]
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)


def yCbCr2rgb(input_im):
    out = torch.zeros_like(input_im)
    for i in range(input_im.shape[0]):
        img_slice = input_im[i, :, :, :].contiguous().float()
        img_slice_y = img_slice[0, :, :]
        img_slice_cb = img_slice[2, :, :]
        img_slice_cr = img_slice[1, :, :]
        # mat = torch.tensor([[1.164, 1.164, 1.164],
        #                     [0, -0.392, 2.017],
        #                     [1.596, -0.813, 0]])
        # bias = torch.tensor([-16.0 / 256.0, -128.0 / 256.0, -128.0 / 256.0])
        # img_slice_y = img_slice_y + bias[0]
        # img_slice_cb = img_slice_cb + bias[1]
        # img_slice_cr = img_slice_cr + bias[2]
        # r = img_slice_y * mat[0][0] + img_slice_cb * mat[1][0] + img_slice_cr * mat[2][0]
        # g = img_slice_y * mat[0][1] + img_slice_cb * mat[1][1] + img_slice_cr * mat[2][1]
        # b = img_slice_y * mat[0][2] + img_slice_cb * mat[1][2] + img_slice_cr * mat[2][2]
        r = img_slice_y + 1.402 * (img_slice_cr - 128 / 256.0)
        g = img_slice_y - 0.34414 * (img_slice_cb - 128 / 256.0) - 0.71414 * (img_slice_cr - 128 / 256.0)
        b = img_slice_y + 1.772 * (img_slice_cb - 128 / 256.0)

        r = r.unsqueeze(0)
        g = g.unsqueeze(0)
        b = b.unsqueeze(0)
        temp = torch.cat([b, g, r], 0)
        temp = temp
        out[i, :, :, :] = temp
    return out


def rgb2yCbCr(input_im):
    out = torch.zeros_like(input_im)
    for i in range(input_im.shape[0]):
        img_slice = input_im[i, :, :, :].contiguous().float()
        img_slice_r = img_slice[2, :, :]
        img_slice_g = img_slice[1, :, :]
        img_slice_b = img_slice[0, :, :]
        # mat = torch.tensor([[0.257, -0.148, 0.439],
        #                     [0.564, -0.291, -0.368],
        #                     [0.098, 0.439, -0.071]])
        # bias = torch.tensor([16.0 / 256.0, 128.0 / 256.0, 128.0 / 256.0])
        # y = img_slice_r * mat[0][0] + img_slice_g * mat[1][0] + img_slice_b * mat[2][0] + bias[0]
        # cb = img_slice_r * mat[0][1] + img_slice_g * mat[1][1] + img_slice_b * mat[2][1] + bias[1]
        # cr = img_slice_r * mat[0][2] + img_slice_g * mat[1][2] + img_slice_b * mat[2][2] + bias[2]
        y = 0.299 * img_slice_r + 0.587 * img_slice_g + 0.114 * img_slice_b
        cb = -0.1687 * img_slice_r - 0.3313 * img_slice_g + 0.5 * img_slice_b + 128 / 256.0
        cr = 0.5 * img_slice_r - 0.4187 * img_slice_g - 0.0813 * img_slice_b + 128 / 256.0
        y = y.unsqueeze(0)
        cb = cb.unsqueeze(0)
        cr = cr.unsqueeze(0)
        temp = torch.cat([y, cr, cb], 0)
        temp = temp
        out[i, :, :, :] = temp
    return out


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor_binary(input):
    input[input < 0.5] = 0
    input[input >= 0.5] = 1
    return input


def resizeAndPadTensor(img, size):
    b, c, h, w = img.shape
    sh, sw = size

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # scale and pad
    scaled_img = torch.zeros(b, c, sh, sw).cuda()
    img = F.interpolate(img, (new_h, new_w), mode='bilinear')
    scaled_img[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img
    # scaled_img = cv.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv.BORDER_CONSTANT, value=padColor)
    return scaled_img, pad_top, pad_left, new_h, new_w


def resize(img, size):
    # usage:
    # images_masked, pad_top, pad_left, new_h, new_w, h, w, resize_flag = resize(images_masked, [64, 64])
    # if resize_flag is True:
    #     mae_preds = resize_back(mae_preds, pad_top, pad_left, new_h, new_w, h, w)

    b, c, h, w = img.shape
    if h != size[0] or w != size[1]:
        img, pad_top, pad_left, new_h, new_w = resizeAndPadTensor(img, size)
        resize_flag = True
    else:
        resize_flag = False
        pad_top, pad_left, new_h, new_w = 0, 0, 0, 0
    return img, pad_top, pad_left, new_h, new_w, h, w, resize_flag


def resize_back(img, pad_top, pad_left, new_h, new_w, h, w):
    img = img[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    img = F.interpolate(img, (h, w), mode='bilinear')
    return img


def metric_iou(prediction, groundtruth):
    intersection = np.logical_and(prediction, groundtruth)
    union = np.logical_or(prediction, groundtruth)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    if np.sum(intersection) + np.sum(union) == 0:
        iou = 1
    return iou


def tensor2cv2(input_tensor: torch.Tensor):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # input_tensor = cv.cvtColor(input_tensor, cv.COLOR_RGB2BGR)
    return input_tensor


def convert(x):
    x = x * 255.
    return x.permute(0, 2, 3, 1).cpu().detach().numpy()


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


def score_metric(pred, target):
    pred_bin, target_bin = convert(pred), convert(target)

    pred_bin = thresholding(pred_bin)
    target_bin = thresholding(target_bin)
    f1 = f1_score(target_bin.flatten(), pred_bin.flatten(), average='macro')
    acc = accuracy_score(target_bin.flatten(), pred_bin.flatten())
    try:
        auc = roc_auc_score(target_bin.flatten(), pred_bin.flatten(), average='macro')
    except ValueError:
        auc = 0
    iou = metric_iou(pred_bin / 255., target_bin / 255.)
    return f1, iou, acc, auc


def backup_code(target_path):
    source_path = ''
    src_file_list = glob(source_path + '*.py')
    for srcfile in src_file_list:
        copy_file(srcfile, target_path)

    src_file_list = glob(source_path + 'models/' + '*.py')
    for srcfile in src_file_list:
        copy_file(srcfile, '{}models/'.format(target_path))

    src_file_list = glob(source_path + 'util/' + '*.py')
    for srcfile in src_file_list:
        copy_file(srcfile, '{}util/'.format(target_path))

    src_file_list = glob(source_path + 'losses/' + '*.py')
    for srcfile in src_file_list:
        copy_file(srcfile, '{}losses/'.format(target_path))


def copy_file(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, dstpath + fname)
        # print("copy %s -> %s" % (srcfile, dstpath + fname))
