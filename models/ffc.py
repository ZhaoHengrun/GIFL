# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import get_activation, BaseDiscriminator
from models.spatial_transform import LearnableSpatialTransformWrapper
from models.squeeze_excitation import SELayer


def padding(img, size):
    b, c, h, w = img.shape  # h = 33
    if h % size != 0 or w % size != 0:
        border_h = size - (h % size)  # border_h = 16-(33%16) = 15
        border_w = size - (w % size)
        pad = nn.ZeroPad2d(padding=(0, border_w, 0, border_h))  # left right up bottom
        img = pad(img)
        padding_flag = True
    else:
        padding_flag = False
    return img, h, w, padding_flag


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
                                              self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
                                              self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)  # if ratio_gin=0.75,in_channels=512, in_cg=384
        in_cl = in_channels - in_cg  # in_cl = 128
        out_cg = int(out_channels * ratio_gout)  # out_cg=384
        out_cl = out_channels - out_cg  # out_cl=128

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=False, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=enable_lfu)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=enable_lfu)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class Lama(nn.Module):
    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 spatial_transform_layers=None,
                 add_out_act='sigmoid', max_features=1024):
        super(Lama, self).__init__()

        self.input_conv = nn.Sequential(nn.ReflectionPad2d(3),
                                        FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                                                   ratio_gin=0, ratio_gout=0, enable_lfu=False)
                                        )

        # ## downsample
        self.downsample = nn.Sequential(FFC_BN_ACT(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer,
                                                   ratio_gin=0, ratio_gout=0, enable_lfu=False),
                                        FFC_BN_ACT(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer,
                                                   ratio_gin=0, ratio_gout=0, enable_lfu=False),
                                        FFC_BN_ACT(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer,
                                                   ratio_gin=0, ratio_gout=0.75, enable_lfu=False),
                                        )

        # ## resnet blocks
        self.res_block = nn.Sequential(FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       )

        self.concat_tuple_layer = ConcatTupleLayer()

        # ## upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            up_norm_layer(ngf * 4), up_activation,
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            up_norm_layer(ngf * 2), up_activation,
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            up_norm_layer(ngf), up_activation,
        )
        self.out_conv = nn.Sequential(

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid()

        )

    def forward(self, input):
        x = self.input_conv(input)
        x = self.downsample(x)  # [8, 384, 32, 32]
        x = self.res_block(x)
        x = self.concat_tuple_layer(x)
        x = self.upsample(x)
        x = self.out_conv(x)
        return x


class HAM(nn.Module):
    def __init__(self, num_feat=64):
        super(HAM, self).__init__()
        self.mask_feat_downsample = nn.Sequential(
            nn.Conv2d(3, num_feat // 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_feat // 8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat // 8, num_feat // 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_feat // 8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.seg_feat_downsample = nn.Sequential(
            nn.Conv2d(3, num_feat // 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_feat // 8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat // 8, num_feat // 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_feat // 8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat // 8, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat // 8, 3, 1, 1)
        self.temporal_attn_x_mask = nn.Conv2d(num_feat // 8, num_feat // 8, 3, 1, 1)
        self.temporal_attn_x_seg = nn.Conv2d(num_feat // 8, num_feat // 8, 3, 1, 1)
        self.temporal_attn_mae_mask = nn.Conv2d(num_feat // 8, num_feat // 8, 3, 1, 1)
        self.temporal_attn_mae_seg = nn.Conv2d(num_feat // 8, num_feat // 8, 3, 1, 1)
        self.corr_increase = nn.Conv2d(num_feat // 8, num_feat, 3, 1, 1)

        self.feat_fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(7, stride=1, padding=3)
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=3)
        self.spatial_attn1 = nn.Conv2d(num_feat * 2, num_feat // 8, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat // 4, num_feat, 1)
        self.spatial_attn_seg = nn.Conv2d(num_feat // 8, num_feat, 3, 1, 1)
        self.spatial_attn_mask = nn.Conv2d(num_feat // 8, num_feat, 3, 1, 1)

        self.spatial_attn_add = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, mae_x, mask, x_seg, mae_pred_seg, patch_mask_64):
        b, c, h, w = x.size()
        mask = F.interpolate(mask, size=(patch_mask_64.shape[2], patch_mask_64.shape[3]), mode='bilinear')
        x_seg = F.interpolate(x_seg, size=(patch_mask_64.shape[2], patch_mask_64.shape[3]), mode='bilinear')

        res_seg = abs(mae_pred_seg - x_seg)
        res_mask = abs(patch_mask_64 - mask)

        mask_feat = torch.cat([mask, patch_mask_64, res_mask], dim=1)
        seg_feat = torch.cat([x_seg, mae_pred_seg, res_seg], dim=1)

        mask_feat = self.mask_feat_downsample(mask_feat)
        seg_feat = self.seg_feat_downsample(seg_feat)

        # temporal attention
        embedding_x = self.temporal_attn1(x)
        embedding_mae = self.temporal_attn2(mae_x)

        embedding_x_mask = self.temporal_attn_x_mask(mask_feat)
        embedding_x_seg = self.temporal_attn_x_seg(seg_feat)
        embedding_x = embedding_x * embedding_x_mask * embedding_x_seg

        embedding_mae_mask = self.temporal_attn_mae_mask(mask_feat)
        embedding_mae_seg = self.temporal_attn_mae_seg(seg_feat)
        embedding_mae = embedding_mae * embedding_mae_mask * embedding_mae_seg

        corr = embedding_x * embedding_mae
        corr = self.corr_increase(corr)
        corr_prob = torch.sigmoid(corr)
        mae_x = mae_x * corr_prob
        # fusion
        x_cat = torch.cat([x, mae_x], dim=1)
        x = self.lrelu(self.feat_fusion(x_cat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(x_cat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = torch.cat([attn_max, attn_avg], dim=1)
        attn = self.lrelu(self.spatial_attn2(attn))

        attn_seg = self.lrelu(self.spatial_attn_seg(seg_feat))
        attn_mask = self.lrelu(self.spatial_attn_mask(mask_feat))

        attn = attn * attn_seg * attn_mask

        attn_add = self.lrelu(self.spatial_attn_add(attn))
        attn = torch.sigmoid(attn)

        x = x * attn + attn_add
        return x


class Hiin(nn.Module):
    def __init__(self, input_nc=8, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 spatial_transform_layers=None,
                 add_out_act='sigmoid', max_features=1024):
        super(Hiin, self).__init__()

        self.mae_downsample = nn.Sequential(
            nn.Conv2d(8, ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.input_conv = nn.Sequential(
            nn.Conv2d(input_nc, ngf * 2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # ## downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.ham = HAM(ngf * 8)

        self.mae_merge = nn.Sequential(
            FFC_BN_ACT(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer,
                       ratio_gin=0, ratio_gout=0.75, enable_lfu=False)
        )
        # ## resnet blocks
        self.res_block = nn.Sequential(FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       FFCResnetBlock(ngf * 8, padding_type=padding_type, activation_layer=activation_layer,
                                                      norm_layer=norm_layer, ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
                                       )

        self.concat_tuple_layer = ConcatTupleLayer()

        # ## upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            up_norm_layer(ngf * 4), up_activation,
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            up_norm_layer(ngf * 2), up_activation,
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            up_norm_layer(ngf), up_activation,
        )

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

        self.seg_conv = nn.Sequential(
            nn.Conv2d(ngf, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, input, mae_pred_input, patch_mask_64):
        b, _, _, _ = input.shape
        x, h, w, padding_flag = padding(input, 16)

        mask = x[:, -1, :, :].unsqueeze(1)
        x_seg = x[:, -2, :, :].unsqueeze(1)
        mae_pred_seg = mae_pred_input[:, -2, :, :].unsqueeze(1)

        x = self.input_conv(input)
        x = self.downsample(x)  # [8, 384, 32, 32]

        mae_x = self.mae_downsample(mae_pred_input)
        x = self.ham(x, mae_x, mask, x_seg, mae_pred_seg, patch_mask_64)
        # x = torch.cat([x, mae_x], dim=1)
        x = self.mae_merge(x)
        x = self.res_block(x)
        x = self.concat_tuple_layer(x)
        x = self.upsample(x)
        seg = self.seg_conv(x)
        img = self.out_conv(x)
        if padding_flag is True:
            img = img[:, :, :h, :w]
            seg = seg[:, :, :h, :w]
        return img, seg


class FFCNLayerDiscriminator(BaseDiscriminator):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, max_features=512,
                 init_conv_kwargs={}, conv_kwargs={}):
        super().__init__()
        self.n_layers = n_layers

        def _act_ctor(inplace=True):
            return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[FFC_BN_ACT(input_nc, ndf, kernel_size=kw, padding=padw, norm_layer=norm_layer,
                                activation_layer=_act_ctor, **init_conv_kwargs)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_features)

            cur_model = [
                FFC_BN_ACT(nf_prev, nf,
                           kernel_size=kw, stride=2, padding=padw,
                           norm_layer=norm_layer,
                           activation_layer=_act_ctor,
                           **conv_kwargs)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = [
            FFC_BN_ACT(nf_prev, nf,
                       kernel_size=kw, stride=1, padding=padw,
                       norm_layer=norm_layer,
                       activation_layer=lambda *args, **kwargs: nn.LeakyReLU(*args, negative_slope=0.2, **kwargs),
                       **conv_kwargs),
            ConcatTupleLayer()
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        feats = []
        for out in act[:-1]:
            if isinstance(out, tuple):
                if torch.is_tensor(out[1]):
                    out = torch.cat(out, dim=1)
                else:
                    out = out[0]
            feats.append(out)
        return act[-1], feats
