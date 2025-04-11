import torch
import numpy as np
import torch.nn.functional as F


class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """

    def __init__(self, weight=1.0):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        # return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1. - pos)) + torch.mean(F.relu(1. + neg)))


class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """

    def __init__(self, weight=1.0):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)


class L1ReconLoss(torch.nn.Module):
    """
    L1 Reconstruction loss for two imgae
    """

    def __init__(self, weight=1):
        super(L1ReconLoss, self).__init__()
        self.weight = weight

    def forward(self, imgs, recon_imgs, masks=None):
        if masks is None:
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
        else:
            # print(masks.view(masks.size(0), -1).mean(1).size(), imgs.size())
            return self.weight * torch.mean(
                torch.abs(imgs - recon_imgs) / masks.view(masks.size(0), -1).mean(1).view(-1, 1, 1, 1))


class ReconLoss(torch.nn.Module):
    """
    Reconstruction loss contain l1 loss, may contain perceptual loss

    """

    def __init__(self, chole_alpha=1.2, cunhole_alpha=1.2, rhole_alpha=1.2, runhole_alpha=1.2):
        super(ReconLoss, self).__init__()
        self.chole_alpha = chole_alpha
        self.cunhole_alpha = cunhole_alpha
        self.rhole_alpha = rhole_alpha
        self.runhole_alpha = runhole_alpha

    def forward(self, imgs, coarse_imgs, recon_imgs, masks):
        masks_viewed = masks.view(masks.size(0), -1)
        return self.rhole_alpha * torch.mean(torch.abs(imgs - recon_imgs) * masks / masks_viewed.mean(1).view(-1, 1, 1, 1)) + \
               self.runhole_alpha * torch.mean(torch.abs(imgs - recon_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1, 1, 1, 1))) + \
               self.chole_alpha * torch.mean(torch.abs(imgs - coarse_imgs) * masks / masks_viewed.mean(1).view(-1, 1, 1, 1)) + \
               self.cunhole_alpha * torch.mean(torch.abs(imgs - coarse_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1, 1, 1, 1)))
