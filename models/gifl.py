import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.dinov2 import DinoV2Encoder, ViT, Transformer, DinoV2EncoderL


def patch_fft(x):
    b, p, d = x.shape  # # [8, 4096, 768] / [8, 3, 896, 896]
    # x = x.reshape(shape=(x.shape[0], 64, 64, -1))
    # print('input_reshape', x.shape)
    x = torch.fft.fft2(x, dim=(-1))
    # print('input_fft_shape', x.shape)
    x = torch.stack((x.real, x.imag), dim=-1)  # [8, 64, 64, 14, 14, 3, 2]
    # print('stack_shape', x.shape)
    x = x.reshape(shape=(b, p, d * 2))  # [8, 64, 64, 14, 14, 3*2]
    # print('reshape', x.shape)
    return x


def patch_ifft(x):
    b, p, d = x.shape  # [8, 4096, 768] / [8, 3*2, 896, 896]
    # print('input_shape', input_shape)
    x = x.reshape(shape=(b, p, d // 2, 2))  # [8, 64, 64, 14, 14, 3]
    # print('input_reshape', x.shape)

    x = torch.complex(x[..., 0], x[..., 1])
    x = torch.fft.ifft2(x, dim=(-1))  # [8, 64, 64, 14, 14, 3]
    # print('ifft_shape', x.shape)
    x = x.reshape(shape=(b, p, d // 2))  # [8, 64, 64, 14, 14, 3*2]
    # print('reshape', x.shape)
    x = x.real
    return x


class GiflB(nn.Module):
    def __init__(self, dim=768, n_classes=196):
        super(FFPredictorB, self).__init__()
        # self.mlp_head = nn.Linear(dim * 2, dim)
        self.transformer = FFTransformer(dim=dim, depth=8, heads=8, dim_head=64, mlp_dim=dim * 4)
        # self.mlp_tail = nn.Linear(dim, n_classes)
        self.decoder = nn.Linear(dim, n_classes)

    def forward(self, x):
        # x = self.mlp_head(x)
        img_feature = self.transformer(x)
        # x = self.mlp_tail(x)
        mask = self.decoder(img_feature)

        mask = mask.reshape(shape=(mask.shape[0], 32, 32, 14, 14, 1))
        mask = torch.einsum('nhwpqc->nchpwq', mask)
        mask = mask.reshape(shape=(mask.shape[0], 1, 448, 448))

        return img_feature, mask


class GiflH(nn.Module):
    def __init__(self, dim=1024, n_classes=196):
        super(GiflH, self).__init__()
        # self.mlp_head = nn.Linear(dim * 2, dim)
        self.transformer = FFTransformer(dim=dim, depth=24, heads=16, dim_head=64, mlp_dim=dim * 4)
        # self.mlp_tail = nn.Linear(dim, n_classes)
        self.decoder = nn.Linear(dim, n_classes)

    def forward(self, x):
        # x = self.mlp_head(x)
        img_feature = self.transformer(x)
        # x = self.mlp_tail(x)
        mask = self.decoder(img_feature)

        mask = mask.reshape(shape=(mask.shape[0], 32, 32, 14, 14, 1))
        mask = torch.einsum('nhwpqc->nchpwq', mask)
        mask = mask.reshape(shape=(mask.shape[0], 1, 448, 448))

        return img_feature, mask





class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        self.mlp_head_f = nn.Linear(dim * 2, dim)
        self.attention_f = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        # self.mlp_tail_f = nn.Linear(dim, dim * 2)

        self.mlp_head_s = nn.Linear(dim, dim * 2)
        self.attention_s = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x_global, x_fft_global):  # dim, dim
        x_fft_local = patch_fft(x_global)  # dim * 2
        x_fft_local = self.mlp_head_f(x_fft_local)  # dim * 2 -> dim
        x_fft = x_fft_global + x_fft_local  # dim
        x_fft = self.attention_f(x_fft)  # dim
        # x_fft = self.mlp_tail_f(x_fft)  # dim -> dim * 2

        x_local = self.mlp_head_s(x_fft_global)  # dim -> dim * 2
        x_local = patch_ifft(x_local)  # dim * 2 -> dim
        x = x_global + x_local  # dim
        x = self.attention_s(x)

        x = x + x_global
        x_fft = x_fft + x_fft_global
        return x, x_fft


class FFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net_s = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.net_f = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, x_fft):
        x = self.net_s(x) + x
        x_fft = self.net_f(x_fft) + x_fft
        return x, x_fft




class FFTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FFeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.mlp_head_f = nn.Linear(dim * 2, dim)
        self.mlp_tail_f = nn.Linear(dim, dim * 2)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_fft = patch_fft(x)  # dim * 2
        x_fft = self.mlp_head_f(x_fft)

        for attn, ff in self.layers:
            x, x_fft = attn(x, x_fft)
            x, x_fft = ff(x, x_fft)

        x_fft = self.mlp_tail_f(x_fft)  # dim -> dim * 2
        x_fft = patch_ifft(x_fft)
        x = x + x_fft
        x = self.feed_forward(x)
        x = self.norm(x)
        return x
