import argparse
import os
import sys
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import cv2 as cv
import torchvision

from dataset import *
from utils import *
from losses.iou import iou_loss
from models.dinov2 import DinoV2Encoder, ViT, Transformer, DinoV2EncoderL
from models.gifl import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_mask", default=[
                                               ],
                    type=list, help="dataset path")
parser.add_argument("--dataset_img", default=[
                                              ], type=list,
                    help="dataset path")

parser.add_argument("--checkpoints_path", default='checkpoints/gifl/', type=str,
                    help="checkpoints path")
parser.add_argument("--resume", default='', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")

parser.add_argument("--start_epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--gpu", default='0', type=str, help="which gpu to use")

opt = parser.parse_args()

n_iter = 0


def main():
    global opt, encoder

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    # torch.cuda.set_device(opt.gpu)
    # device = torch.device('cuda:{}'.format(opt.gpu))
    device = torch.device('cuda:0')

    print("> Loading datasets")
    train_set = TrainDataset(img_path_lis=opt.dataset_img, mask_path_lis=opt.dataset_mask, resize=448)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                                      shuffle=True, num_workers=opt.threads)

    print("> Building model")
    encoder = DinoV2EncoderL()
    encoder.eval()
    predictor = GiflH()

    loss_fun_feature = nn.MSELoss()
    loss_fun_mask = nn.BCEWithLogitsLoss()

    print("> Setting GPU")

    encoder = nn.DataParallel(encoder)
    encoder = encoder.to(device)
    predictor = nn.DataParallel(predictor)
    predictor = predictor.to(device)

    loss_fun_feature = loss_fun_feature.to(device)
    loss_fun_mask = loss_fun_mask.to(device)

    print("> Setting Optimizer")
    optimizer = optim.Adam(predictor.parameters(), lr=opt.lr)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print('==> load pretrained model from:{}'.format(opt.resume))
            checkpoint = torch.load(opt.resume)
            predictor.load_state_dict(checkpoint['predictor'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            opt.start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("> Training")
    if not os.path.exists(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)
        print('create path:', opt.checkpoints_path)
        os.makedirs('{}{}'.format(opt.checkpoints_path, 'img/'))
        os.makedirs('{}{}'.format(opt.checkpoints_path, 'code/'))

    backup_code('{}{}'.format(opt.checkpoints_path, 'code/'))

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, loss_fun_feature,
              loss_fun_mask, epoch, encoder, predictor)
        save_checkpoint(optimizer, predictor, epoch)


def train(training_data_loader, optimizer, loss_fun_feature,
          loss_fun_mask, epoch, encoder, predictor):
    global n_iter

    avr_loss_feature = 0
    # avr_loss_pre_mask = 0
    avr_loss_mask = 0

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    predictor.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        n_iter = iteration
        input, mask, h, w = batch[0], batch[1], batch[2], batch[3]

        input = input.cuda()
        mask = mask.cuda()

        masked_img = input * (1 - mask)

        with torch.no_grad():
            input_feature = encoder(input)
            target_feature = encoder(masked_img)

        #  ------------------------------feature loss------------------------------
        img_feature, mask_pred = predictor(input_feature)  # [8, 4096, 768]

        loss_feature = loss_fun_feature(img_feature, target_feature)
        #  -------------------------------mask loss-----------------------------------

        loss_mask = loss_fun_mask(mask_pred, mask)
        loss_mask += iou_loss(mask_pred, mask)

        loss = loss_feature * 10 + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 1000 == 0:
            save_image_tensor(input[0, :, :, :].unsqueeze(0),
                              '{}img/E[{}]_input.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(mask[0, :, :, :].unsqueeze(0),
                              '{}img/E[{}]_target.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(mask_pred[0, :, :, :].unsqueeze(0),
                              '{}img/E[{}]_mask_pred.png'.format(opt.checkpoints_path, epoch))

        avr_loss_feature += loss_feature.item()
        avr_loss_mask += loss_mask.item()

        sys.stdout.write(
            "> E[{}]({}/{}): L_p: [{:.4f}], L_m:[{:.4f}]\r"
            .format(epoch, iteration, len(training_data_loader),
                    loss_feature.item(), loss_mask.item()))

    avr_loss_feature = avr_loss_feature / len(training_data_loader)
    avr_loss_mask = avr_loss_mask / len(training_data_loader)

    with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
        f.write('E[{}], L_p: [{:.4f}], L_m:[{:.4f}]\n'
                .format(epoch, avr_loss_feature, avr_loss_mask))

    torch.cuda.empty_cache()


def save_checkpoint(optimizer, predictor, epoch):
    global save_flag

    checkpoint = {
        "predictor": predictor.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, opt.checkpoints_path + "last_model.pth")

    if (epoch % 10) == 0:
        torch.save(checkpoint, opt.checkpoints_path + "model_epoch_{}.pth".format(epoch))
        print("Checkpoint saved to {}".format(opt.checkpoints_path))


if __name__ == "__main__":
    main()
