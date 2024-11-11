import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange


EPSILON = 1e-10


def var(x, dim=0):
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


class MultConst(nn.Module):
    def forward(self, input):
        return 255*input


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    """1*1 conv before the output"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)


    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


class h4_PT_hd3_eval(torch.nn.Module):
    def __init__(self):
        super(h4_PT_hd3_eval, self).__init__()
        self.down4 = nn.MaxPool2d(4, 4, ceil_mode=True)

    def forward(self, x1, x2):
        x2 = self.down4(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x2[3] - shape_x1[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:

            top_bot = shape_x2[2] - shape_x1[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [-left, -right, -top, -bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


class h2_PT_hd3_eval(torch.nn.Module):
    def __init__(self):
        super(h2_PT_hd3_eval, self).__init__()
        self.down2 = nn.MaxPool2d(2, 2, ceil_mode=True)


    def forward(self, x1, x2):
        x2 = self.down2(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x2[3] - shape_x1[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x2[2] - shape_x1[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [-left, -right, -top, -bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim) ##

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv) # Divided into multiple heads

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots) # softmax operation

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) ##
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class aggregation(nn.Module):

    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth): # the number of encoder is depth. (PreNorm is the LayerNorm; Attention is MSA; FeedForward is the MLP.)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=1, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        self.dim = dim
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # initialize the parameter of clc_token
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.convd1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))



    def forward(self, img):

        x = self.to_patch_embedding(img)  # [B,256,256]
        b, n, _ = x.shape

        x = self.transformer(x)
        x = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, h=16, c=1)(x)  # [B,1,256,256]

        return x



class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out



class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = ConvLayer(2*channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')
        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init = self.conv_fusion(f_cat)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi)
        out = torch.cat([out_ir, out_vi], 1)
        out = self.bottelblock(out)
        out = f_init + out
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
## 空间注意力机制
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial

class FusionBlock_res_cross(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res_cross, self).__init__()
        ws = [3, 3, 3, 3]
        kernel_size = 3
        stride = 1
        block1 = DenseBlock_light
        self.conv_fusion = ConvLayer(2*channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(channels, channels)
        self.layer2 = DoubleConv(32, 48)
        self.conv_out = ConvLayer(channels, 1, 1, stride)
        # self.up = nn.Upsample(scale_factor=2)


        self.DBU1_3 = block1(channels, 1, kernel_size, 1)
        self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=3, heads=8, mlp_dim=512, dropout=0.1,
                               emb_dropout=0.1)


    def forward(self, x_ir, x_vi):

        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init = self.conv_fusion(f_cat) + x_ir + x_vi

        x_avg = (x_ir+ x_vi) / 2
        x_max = torch.max(x_ir, x_vi)
        out = torch.cat([x_avg, x_max], 1)
        out = self.bottelblock(out)
        out = f_init + out

        return out

class FusionBlock_RF_cross(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_RF_cross, self).__init__()
        ws = [3, 3, 3, 3]
        kernel_size = 3
        stride = 1
        block1 = DenseBlock_light
        self.conv_fusion = ConvLayer(2*channels, channels, ws[index], 1)
        block2 = []
        block2 += [ConvLayer(2 * channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock2 = nn.Sequential(*block2)

        block3 = []
        block3 += [ConvLayer(3 * channels, 2 * channels, ws[index], 1),
                   ConvLayer(2 * channels, channels, ws[index], 1),
                   ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock3 = nn.Sequential(*block3)
        self.conv_fusion3 = ConvLayer(3 * channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(channels, channels)
        self.layer2 = DoubleConv(32, 48)
        self.conv_out = ConvLayer(channels, 1, 1, stride)



        self.DBU1_3 = block1(channels, 1, kernel_size, 1)
        self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=3, heads=8, mlp_dim=512, dropout=0.1,
                               emb_dropout=0.1)


    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')
        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init_1 = self.bottelblock2(f_cat)
        f_init_2 = self.conv_ir (f_init_1)
        f_init_3 = self.conv_ir(f_init_2)
        f_cat_out_pre = torch.cat([f_init_1, f_init_2,f_init_3], 1)
        f_cat_out = self.bottelblock3(f_cat_out_pre)
        f_w = torch.softmax(f_cat_out,dim=0)
        out = x_ir*f_w+x_vi*(1-f_w)

        return out


class FusionBlock_CAF_cross(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_CAF_cross, self).__init__()

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())
        self.channel_attention_1 = nn.Sequential(
            nn.Conv2d(2*channels, channels, 1, padding=0),
            nn.Conv2d(channels, channels, 1, padding=0),
            nn.ReLU(inplace=True))

        self.channel_attention_2 = nn.Sequential(nn.Sigmoid())



        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.cross_conv = nn.Conv2d(32 * 2, 32, 1, padding=0)
        self.cross_conv = nn.Conv2d(channels * 2, channels, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

        ws = [3, 3, 3, 3]
        kernel_size = 3
        stride = 1
        block1 = DenseBlock_light
        self.conv_fusion = ConvLayer(2*channels, channels, ws[index], 1)
        block2 = []
        block2 += [ConvLayer(channels, channels, ws[index], 0),
                  ConvLayer(channels, channels, ws[index], 0),
                  ConvLayer(channels, channels, ws[index], 0)]
        self.bottelblock2 = nn.Sequential(*block2)

        block3 = []
        block3 += [ConvLayer(3 * channels, 2 * channels, ws[index], 1),
                   ConvLayer(2 * channels, channels, ws[index], 1),
                   ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock3 = nn.Sequential(*block3)
        self.conv_fusion3 = ConvLayer(3 * channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(channels, channels)
        self.layer2 = DoubleConv(32, 48)
        self.conv_out = ConvLayer(channels, 1, 1, stride)
        # self.up = nn.Upsample(scale_factor=2)


        self.DBU1_3 = block1(channels, 1, kernel_size, 1)
        self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=3, heads=8, mlp_dim=512, dropout=0.1,
                               emb_dropout=0.1)


    def forward(self, x_ir, x_vi):

        x3_r = x_ir
        x3_d = x_vi


        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        f_cat_ir_vi = torch.cat([self.squeeze_rgb(x3_r),self.squeeze_rgb(x3_d)],1)
        f_cat_ir_vi1 = self.channel_attention_1(f_cat_ir_vi)
        f_cat_ir_vi2 = self.channel_attention_2(f_cat_ir_vi1)
        f_cat_ir_co = x3_r * f_cat_ir_vi2.expand_as(x3_r)
        f_cat_vi_co = x3_d * f_cat_ir_vi2.expand_as(x3_r)



        SCA_3_o = x3_r * SCA_ca.expand_as(x3_r)

        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        SCA_3d_o = x3_d * SCA_d_ca.expand_as(x3_d)

        Co_ca3 = torch.softmax(SCA_ca + SCA_d_ca, dim=1)

        SCA_3_co = x3_r * Co_ca3.expand_as(x3_r)
        SCA_3d_co = x3_d * Co_ca3.expand_as(x3_d)

        CR_fea3_rgb = SCA_3_o + SCA_3_co
        CR_fea3_d = SCA_3d_o + SCA_3d_co

        CR_fea3_rgb = SCA_3_o + f_cat_ir_co
        CR_fea3_d = SCA_3d_o + f_cat_vi_co

        CR_fea3 = torch.cat([CR_fea3_rgb, CR_fea3_d], dim=1)
        out = self.bottelblock(CR_fea3)


        return out


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()


        out = a_w * a_h

        return out


class FusionBlock_CCAF_cross(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_CCAF_cross, self).__init__()

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())
        self.channel_attention_1 = nn.Sequential(
            nn.Conv2d(2*channels, channels, 1, padding=0),
            nn.Conv2d(channels, channels, 1, padding=0),
            nn.ReLU(inplace=True))

        self.channel_attention_2 = nn.Sequential(nn.Sigmoid())



        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.cross_conv = nn.Conv2d(channels * 2, channels, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

        ws = [3, 3, 3, 3]
        kernel_size = 3
        stride = 1
        block1 = DenseBlock_light
        self.conv_fusion = ConvLayer(2*channels, channels, ws[index], 1)
        block2 = []
        block2 += [ConvLayer(channels, channels, ws[index], 0),
                  ConvLayer(channels, channels, ws[index], 0),
                  ConvLayer(channels, channels, ws[index], 0)]
        self.bottelblock2 = nn.Sequential(*block2)

        block3 = []
        block3 += [ConvLayer(3 * channels, 2 * channels, ws[index], 1),
                   ConvLayer(2 * channels, channels, ws[index], 1),
                   ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock3 = nn.Sequential(*block3)
        self.conv_fusion3 = ConvLayer(3 * channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)
        self.ca = CoordAtt(channels, channels, reduction=32)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)
        self.inc = DoubleConv(1, 16)
        self.layer1 = DoubleConv(channels, channels)
        self.layer2 = DoubleConv(32, 48)
        self.conv_out = ConvLayer(channels, 1, 1, stride)
        # self.up = nn.Upsample(scale_factor=2)


        self.DBU1_3 = block1(channels, 1, kernel_size, 1)
        self.transformer = ViT(image_size=256, patch_size=16, dim=256, depth=3, heads=8, mlp_dim=512, dropout=0.1,
                               emb_dropout=0.1)


    def forward(self, x_ir, x_vi):

        CCAF_ir = self.ca(x_ir)
        CCAF_vi = self.ca(x_vi)
        Co_CCAF = torch.softmax(CCAF_ir + CCAF_vi, dim=1)

        CCAF_ir_o = x_ir * CCAF_ir+(1-CCAF_ir) * x_vi
        CCAF_vi_o = x_vi * CCAF_vi+(1-CCAF_vi) * x_ir
        CCAF_ir_co = x_ir * Co_CCAF
        CCAF_vi_co = x_vi * Co_CCAF
        CR_fea3 = torch.cat([CCAF_ir_o+CCAF_ir_co, CCAF_vi_o+CCAF_vi_co], dim=1)
        out = self.bottelblock(CR_fea3)

        return out

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)



def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out+x, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)

        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet18-5c106cde.pth'), strict=False)


# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)

        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)


# Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SAM, self).__init__()
        self.conv_atten = conv3x3(2, 1)
        self.conv = conv3x3(in_chan, out_chan)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten)
        out = F.relu(self.bn(self.conv(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Boundary Refinement Module
class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        # x = torch.cat([x_1, x_edge], dim=1)
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Fusion network, 4 groups of features
class Fusion_network(nn.Module):
    def __init__(self, nC, fs_type):
        super(Fusion_network, self).__init__()
        self.fs_type = fs_type


        self.fusion_block1 = FusionBlock_res(nC[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]

# ------------------------CCAF cross fusion ----------------------------------
class Fusion_network_CCAF_cross(nn.Module):
    def __init__(self, nC, fs_type):
        super(Fusion_network_CCAF_cross, self).__init__()
        self.fs_type = fs_type


        self.fusion_block1 = FusionBlock_CCAF_cross(nC[0], 0)
        self.fusion_block2 = FusionBlock_CCAF_cross(nC[1], 1)
        self.fusion_block3 = FusionBlock_CCAF_cross(nC[2], 2)
        self.fusion_block4 = FusionBlock_CCAF_cross(nC[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])


        return [f1_0, f2_0, f3_0, f4_0]




# ------------------------ salient detection ----------------------------------
class ir_salient_detection(nn.Module):
    def __init__(self, nC, fs_type):
        super(ir_salient_detection, self).__init__()

        self.fs_type = fs_type
        self.conv1 = nn.Conv2d(nC[3], nC[3], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(nC[3])

        block = BasicBlock
        self.bkbone = ResNet(block, [2, 2, 2, 2])

        self.path1_1 = nn.Sequential(
            conv1x1(512 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.path1_2 = nn.Sequential(
            conv1x1(512 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.path1_3 = nn.Sequential(
            conv1x1(256 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.path2 = SAM(128 * block.expansion, 64)

        self.path3 = nn.Sequential(
            conv1x1(64 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.fuse1_1 = FFM(64)
        self.fuse1_2 = FFM(64)
        self.fuse3 = FFM(64)



    def forward(self, en_ir, en_vi):

        out1 = F.relu(self.bn1(self.conv1(en_ir[3])), inplace=True)
        out1 = F.max_pool2d(out1,kernel_size=3,stride=2,padding=1)

        path1_1 = F.avg_pool2d(en_ir[3], en_ir[3].size()[2:])
        path1_1 = self.path1_1(path1_1)
        path1_1 = F.interpolate(path1_1, size=en_ir[3].size()[2:], mode='bilinear', align_corners=True)  # 1/32
        path1_2 = F.relu(self.path1_2(en_ir[3]), inplace=True)  # 1/32
        path1_2 = self.fuse1_1(path1_1, path1_2)  # 1/32
        path1_2 = F.interpolate(path1_2, size=en_ir[2].size()[2:], mode='bilinear', align_corners=True)  # 1/16

        path1_3 = F.relu(self.path1_3(en_ir[2]), inplace=True)  # 1/16
        path1 = self.fuse1_2(path1_2, path1_3)  # 1/16

        path2_1 = F.interpolate(path1, size=en_ir[1].size()[2:], mode='bilinear', align_corners=True)  # 1/16

        path2_2 = F.relu(self.path1_3(en_ir[1]), inplace=True)  # 1/16
        path2 = self.fuse1_2(path2_1, path2_2)

        path3_1 = F.interpolate(path2, size=en_ir[0].size()[2:], mode='bilinear', align_corners=True)  # 1/16

        path3_2 = F.relu(self.path1_3(en_ir[0]), inplace=True)  # 1/16
        path4 = self.fuse1_2(path3_1, path3_2)

        return [path4]

class CTDNet(nn.Module):
    def __init__(self, cfg):
        super(CTDNet, self).__init__()
        self.cfg = cfg
        block = BasicBlock
        self.bkbone = ResNet(block, [2, 2, 2, 2])

        self.path1_1 = nn.Sequential(
            conv1x1(512 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.path1_2 = nn.Sequential(
            conv1x1(512 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )
        self.path1_3 = nn.Sequential(
            conv1x1(256 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.path2 = SAM(128 * block.expansion, 64)

        self.path3 = nn.Sequential(
            conv1x1(64 * block.expansion, 64),
            nn.BatchNorm2d(64)
        )

        self.fuse1_1 = FFM(64)
        self.fuse1_2 = FFM(64)
        self.fuse12 = CAM(64)
        self.fuse3 = FFM(64)
        self.fuse23 = BRM(64)

        self.head_1 = conv3x3(64, 1, bias=True)
        self.head_2 = conv3x3(64, 1, bias=True)
        self.head_3 = conv3x3(64, 1, bias=True)
        self.head_4 = conv3x3(64, 1, bias=True)
        self.head_5 = conv3x3(64, 1, bias=True)
        self.head_edge = conv3x3(64, 1, bias=True)

        self.initialize()

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape
        l1, l2, l3, l4, l5 = self.bkbone(x)

        path1_1 = F.avg_pool2d(l5, l5.size()[2:])
        path1_1 = self.path1_1(path1_1)
        path1_1 = F.interpolate(path1_1, size=l5.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        path1_2 = F.relu(self.path1_2(l5), inplace=True)                                            # 1/32
        path1_2 = self.fuse1_1(path1_1, path1_2)                                                    # 1/32
        path1_2 = F.interpolate(path1_2, size=l4.size()[2:], mode='bilinear', align_corners=True)   # 1/16

        path1_3 = F.relu(self.path1_3(l4), inplace=True)                                            # 1/16
        path1 = self.fuse1_2(path1_2, path1_3)                                                      # 1/16
        # path1 = F.interpolate(path1, size=l3.size()[2:], mode='bilinear', align_corners=True)

        path2 = self.path2(l3)                                                                      # 1/8
        path12 = self.fuse1_3(path1, path2)                                                          # 1/8
        path12 = F.interpolate(path12, size=l2.size()[2:], mode='bilinear', align_corners=True)     # 1/4

        path3_1 = F.relu(self.path3(l2), inplace=True)                                              # 1/4
        path3_2 = F.interpolate(path1_2, size=l2.size()[2:], mode='bilinear', align_corners=True)   # 1/4
        path3 = self.fuse1_4(path3_1, path3_2)                                                        # 1/4

        path_out = self.fuse23(path12, path3)                                                       # 1/4

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)
        logits_edge = F.interpolate(self.head_edge(path3), size=shape, mode='bilinear', align_corners=True)

        if self.cfg.mode == 'train':
            logits_2 = F.interpolate(self.head_2(path12), size=shape, mode='bilinear', align_corners=True)
            logits_3 = F.interpolate(self.head_3(path1), size=shape, mode='bilinear', align_corners=True)
            logits_4 = F.interpolate(self.head_4(path1_2), size=shape, mode='bilinear', align_corners=True)
            logits_5 = F.interpolate(self.head_5(path1_1), size=shape, mode='bilinear', align_corners=True)
            return logits_1, logits_edge, logits_2, logits_3, logits_4, logits_5
        else:
            return logits_1, logits_edge

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)

# ---------------------- fusion meta ------------------------------------
class Pos2Weight(nn.Module):
    def __init__(self, inC, outC, kernel_size=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.outC = outC
        self.kernel_size=kernel_size
        self.meta_block=nn.Sequential(
            nn.Linear(3, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.kernel_size*self.kernel_size*self.inC*self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)

        return output

class MetaLearner(nn.Module):
    def __init__(self, args):
        super(MetaLearner, self).__init__()
        self.P2W = Pos2Weight(inC=args.meta_C, outC=args.meta_C)

    def repeat_x(self, x, r_int):
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)
        x = torch.cat([x] * r_int, 3)
        x = torch.cat([x] * r_int, 5).permute(0, 3, 5, 1, 2, 4)

        return torch.reshape(x, (-1, C, H, W))

    def forward(self, x, r, pos_mat, mask, HRsize):
        # torch.cuda.empty_cache()
        scale_int = math.ceil(r)
        outC = x.size(1)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_x = self.repeat_x(x, scale_int)
        cols = nn.functional.unfold(up_x, kernel_size=3, padding=1)
        cols = torch.reshape(cols, (cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2), 1)).permute(0, 1, 3, 4, 2)
        local_weight = torch.reshape(local_weight, (x.size(2), scale_int, x.size(3), scale_int, -1, outC)).permute(1,3,0,2,4,5)
        local_weight = torch.reshape(local_weight, (scale_int ** 2, x.size(2) * x.size(3), -1, outC))
        out = torch.matmul(cols, local_weight)
        out = out.permute(0, 1, 4, 2, 3)
        out = torch.reshape(out, (x.size(0), scale_int, scale_int, outC, x.size(2), x.size(3))).permute(0, 3, 4, 1, 5, 2)
        out = torch.reshape(out, (x.size(0), outC, scale_int * x.size(2), scale_int * x.size(3)))

        re_sr = torch.masked_select(out, mask)
        re_sr = torch.reshape(re_sr, (x.size(0), outC, HRsize[0], HRsize[1]))
        torch.cuda.empty_cache()

        return re_sr

class FusionBlock_res_meta(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res_meta, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = ConvLayer(2*channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')
        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init = self.conv_fusion(f_cat)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi) # 原来的代码有问题，写成了conv_ir，现在重新训练
        out = torch.cat([out_ir, out_vi], 1)
        out = self.bottelblock(out)
        out = f_init + out
        return out



class Fusion_ADD(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = en_ir + en_vi
        return temp


class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp


class Fusion_MAX(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = torch.max(en_ir, en_vi)
        return temp


class Fusion_SPA(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        spatial_type = 'mean'
        # calculate spatial attention
        spatial1 = spatial_attention(en_ir, spatial_type)
        spatial2 = spatial_attention(en_vi, spatial_type)
        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * en_ir + spatial_w2 * en_vi
        return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


# fuison strategy based on nuclear-norm (channel attention form NestFuse)
class Fusion_Nuclear(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        # calculate channel attention
        global_p1 = nuclear_pooling(en_ir)
        global_p2 = nuclear_pooling(en_vi)

        # get weight map
        global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

        tensor_f = global_p_w1 * en_ir + global_p_w2 * en_vi
        return tensor_f


# sum of S V for each chanel
def nuclear_pooling(tensor):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors


# Fusion strategy, two type
class Fusion_strategy(nn.Module):
    def __init__(self, fs_type: object) -> object:
        super(Fusion_strategy, self).__init__()
        self.fs_type = fs_type
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_spa = Fusion_SPA()
        self.fusion_nuc = Fusion_Nuclear()

    def forward(self, en_ir, en_vi):
        if self.fs_type is 'add':
            fusion_operation = self.fusion_add
        elif self.fs_type is 'avg':
            fusion_operation = self.fusion_avg
        elif self.fs_type is 'max':
            fusion_operation = self.fusion_max
        elif self.fs_type is 'spa':
            fusion_operation = self.fusion_spa
        elif self.fs_type is 'nuclear':
            fusion_operation = self.fusion_nuc

        f1_0 = fusion_operation(en_ir[0], en_vi[0])
        f2_0 = fusion_operation(en_ir[1], en_vi[1])
        f3_0 = fusion_operation(en_ir[2], en_vi[2])
        f4_0 = fusion_operation(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]


# NestFuse network - light, no desnse
class NestFuse_light2_nodense(nn.Module):
    def __init__(self, nb_filter: object, input_nc: object = 1, output_nc: object = 1, deepsupervision: object = True) -> object:
        super(NestFuse_light2_nodense, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()
        self.h4_PT_hd3_eval = h4_PT_hd3_eval()
        self.h2_PT_hd3_eval = h2_PT_hd3_eval()

        # encoder

        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        # decoder
        self.h4_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.DBV1_1 = block(nb_filter[0] + nb_filter[1] + nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.DBV1_2 = block(nb_filter[0] + nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DBV1_3 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)

        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)


        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            # self.conv4 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def decoder_train(self, f_en):
        x1_1 = self.DBV1_1(torch.cat([self.h4_PT_hd3(f_en[0]), self.h2_PT_hd3(f_en[1]), f_en[2], self.up(f_en[3])], 1))
        x1_2 = self.DBV1_2(torch.cat([self.h2_PT_hd3(f_en[0]), f_en[1], self.up(x1_1)], 1))
        x1_3 = self.DBV1_3(torch.cat([f_en[0], self.up(x1_2)], 1))


        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):

        x1_1 = self.DBV1_1(torch.cat([self.h4_PT_hd3_eval(f_en[2], f_en[0]), self.h2_PT_hd3_eval(f_en[2], f_en[1]), f_en[2],self.up_eval(f_en[2], f_en[3])], 1))
        x1_2 = self.DBV1_2(torch.cat([self.h2_PT_hd3_eval(f_en[1], f_en[0]), f_en[1], self.up_eval(f_en[1], x1_1)], 1))
        x1_3 = self.DBV1_3(torch.cat([f_en[0], self.up_eval(f_en[0], x1_2)], 1))


        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]


