# from __future__ import print_function
import argparse
from os import listdir
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
from torchvision import transforms
from dataset import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
from einops import rearrange
import pyiqa
from sklearn.metrics import f1_score
from models.dinov2 import DinoV2Encoder, ViT, Transformer, DinoV2EncoderL
from torch_kmeans import KMeans
from torch_kmeans.utils.distances import CosineSimilarity
from models.gifl import *
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

# Training settings
parser = argparse.ArgumentParser(description='Test')
parser.add_argument("--test_length", default=10, type=int, help="how many img to test, -1 for all")
parser.add_argument("--dataset_test_input", default=[

],
                    type=list, help="dataset path")
parser.add_argument("--dataset_test_mask", default=[

], type=list, help="mask path")
parser.add_argument('--output_path', default='output/gifl/', type=str,
                    help='where to save the output image')
parser.add_argument('--model_path', type=str, default='checkpoints/gifl/last_model.pth',
                    help='model file to use')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--save_img', default=True, action='store_true', help='save img')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test():
    print('datasets:{}'.format(opt.dataset_test_input))
    print('loading model')
    encoder = DinoV2EncoderL()
    encoder.eval()
    encoder = nn.DataParallel(encoder)
    encoder = encoder.cuda()

    predictor = GiflH()
    predictor = nn.DataParallel(predictor)
    checkpoint = torch.load(opt.model_path)
    predictor.load_state_dict(checkpoint['predictor'])
    predictor.eval()
    predictor = predictor.cuda()

    for index in range(len(opt.dataset_test_input)):
        dataset_test_input = opt.dataset_test_input[index]
        dataset_test_mask = opt.dataset_test_mask[index]
        test_name = dataset_test_input.split('/')[-2]

        test_set = ValidDataset(dataset_test_input, dataset_test_mask, 448, opt.test_length)
        test_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
        print('test_name:', test_name)
        print(len(test_data_loader), 'test images')

        make_dir(opt.output_path)
        test_sub_path = '{}{}/'.format(opt.output_path, test_name)
        make_dir(test_sub_path)

        input_path = '{}input/'.format(test_sub_path)
        pred_path = '{}pred/'.format(test_sub_path)
        mask_path = '{}mask/'.format(test_sub_path)

        if opt.save_img is True:
            make_dir(input_path)
            make_dir(pred_path)
            make_dir(mask_path)

        count = 0
        f1_sum = 0
        iou_sum = 0
        acc_sum = 0
        auc_sum = 0

        y_true, y_pred = [], []

        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 1):
                count += 1
                input, mask, h, w, input_name_list = batch[0], batch[1], batch[2], batch[3], batch[4]
                input_name = '{}.png'.format(input_name_list[0].split('.')[0])
                print('processing:[{}/{}][{}]'.format(count, len(test_data_loader), input_name))

                input = input.cuda()
                mask = mask.cuda()

                with torch.no_grad():
                    input_feature = encoder(input)

                img_feature, mask_pred = predictor(input_feature)

                f1_score, iou_score, acc_score, auc_score = score_metric(mask_pred, mask)

                f1_sum += f1_score
                iou_sum += iou_score
                acc_sum += acc_score
                auc_sum += auc_score

                if test_name == 'test_img':
                    target_label = mask.view(mask.shape[0], -1)
                    target_label = torch.max(target_label)
                    target_label = torch.where(target_label > 0.5, torch.ones_like(target_label),
                                               torch.zeros_like(target_label))

                    pred_label = mask_pred.view(mask_pred.shape[0], -1)
                    pred_label = torch.max(pred_label)
                    pred_label = torch.where(pred_label > 0.5, torch.ones_like(pred_label),
                                             torch.zeros_like(pred_label))
                    print('target:{}, pred:{}'.format(target_label.item(), pred_label.item()))
                    y_pred.extend(pred_label.flatten().tolist())
                    y_true.extend(target_label.flatten().tolist())

                h, w = h.squeeze(0), w.squeeze(0)
                if opt.save_img is True:
                    input = F.interpolate(input, size=(h, w))
                    mask = F.interpolate(mask, size=(h, w))
                    mask_pred = F.interpolate(mask_pred, size=(h, w))

                    save_image_tensor(input, '{}{}'.format(input_path, input_name))
                    save_image_tensor(mask_pred, '{}{}'.format(pred_path, input_name))
                    save_image_tensor(mask, '{}{}'.format(mask_path, input_name))

        f1_avr = f1_sum / len(test_data_loader)
        iou_avr = iou_sum / len(test_data_loader)
        acc_avr = acc_sum / len(test_data_loader)
        auc_avr = auc_sum / len(test_data_loader)

        print('f1:[{}]'.format(f1_avr))
        print('iou:[{}]'.format(iou_avr))
        print('acc:[{}]'.format(acc_avr))
        print('auc:[{}]'.format(auc_avr))
        with open("{}results.txt".format(opt.output_path), "a") as f:
            f.write(
                '{}: f1: {:.4f} , iou: {:.4f} , acc: {:.4f} , auc: {:.4f} \n'
                .format(test_name, f1_avr, iou_avr, acc_avr, auc_avr))

        if test_name == 'test_img':
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            acc_l = accuracy_score(y_true, y_pred)

            print('acc_l:[{:.4f}]'.format(acc_l))

            with open("{}results.txt".format(opt.output_path), "a") as f:
                f.write(
                    '{}: acc_l: {:.4f} \n'
                    .format(test_name, acc_l))

    print('finish')


if __name__ == "__main__":
    test()
