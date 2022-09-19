import sys

import torchvision

sys.path.append('D:\\Document\\PycharmProjects\\EarlyDetection\\src\\net\\GlobalLocal')
sys.path.append('/root/autodl-octnet/dl/src/net/GlobalLocal')

import math
import torch
import torch.nn as nn
import vgg as vnet
from src.net.octnet.octnet import octnet
from visualizer import get_local

class GlobalAttention(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.project_query = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.project_key = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.project_value = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.project_out = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, locx, glox):
        locx_query_mix = self.project_query(locx)  # b*512*2*2
        glox_key_mix = self.project_key(glox)  # b*512*16*16
        glox_value_mix = self.project_value(glox)  # b*512*16*16

        jQ = torch.flatten(locx_query_mix, 2)
        jK = torch.flatten(glox_key_mix, 2)
        jV = torch.flatten(glox_value_mix, 2)

        attention_scores = torch.matmul(jQ.transpose(1, 2), jK)
        attention_scores = torch.div(attention_scores, math.sqrt(self.hidden_size))
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, jV.transpose(1, 2)).permute(0, 2, 1). \
            view(-1, self.hidden_size, locx.shape[2], locx.shape[2])  # 1*512*4

        attention_output = self.project_out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output


class convBlock(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=3, padding=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class Feedforward(nn.Module):
    def __init__(self, inplace, outplace):
        super().__init__()

        self.conv1 = convBlock(inplace, outplace, kernel_size=1, padding=0)
        self.conv2 = convBlock(outplace, outplace, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GlobalLocalTransformer(nn.Module):

    def __init__(self, inplace,
                 nblock=6,
                 drop_rate=0.5,
                 backbone='vgg8'):
        super().__init__()

        self.nblock = nblock
        hidden_size = -1
        if backbone == 'vgg8':
            self.global_feat = vnet.VGG8(inplace)
            self.local_feat = vnet.VGG8(inplace)
            hidden_size = 512
        elif backbone == 'vgg16':
            self.global_feat = vnet.VGG16(inplace)
            self.local_feat = vnet.VGG16(inplace)
            hidden_size = 512
        elif backbone == 'octnet':
            self.global_feat = octnet()
            self.local_feat = octnet()
            hidden_size = 256
        elif backbone == 'efficientnet_b7':
            net = torchvision.models.efficientnet_b7(pretrained=True)
            net.features[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.global_feat = net.features
            self.local_feat = net.features
            hidden_size = 2560
        elif backbone == 'efficientnet_b0':
            net = torchvision.models.efficientnet_b0(pretrained=True)
            net.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.global_feat = net.features
            self.local_feat = net.features
            hidden_size = 1280
        elif backbone == 'inception':
            net = torchvision.models.inception_v3(pretrained=True)
            net.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
            self.global_feat = net
            self.local_feat = net

        self.attnlist = nn.ModuleList()
        self.fftlist = nn.ModuleList()

        for n in range(nblock):
            atten = GlobalAttention(
                hidden_size=hidden_size,
                transformer_dropout_rate=drop_rate)
            self.attnlist.append(atten)

            fft = Feedforward(inplace=hidden_size,
                              outplace=hidden_size)
            self.fftlist.append(fft)

        self.avg = nn.AdaptiveAvgPool2d(1)
        out_hidden_size = hidden_size

        self.gloout = nn.Linear(out_hidden_size, 2)
        self.locout = nn.Linear(out_hidden_size, 2)

    @get_local('xloc')
    def forward(self, x):
        xinput = x[0]
        patches = x[1]
        outlist = []
        # Global path
        xglo = self.global_feat(xinput)  # backbone生成的feature_map。四个维度，sample_n * channel_n * H * W
        xgloout = self.gloout(torch.flatten(self.avg(xglo), 1))
        outlist.append(xgloout)
        # Local path
        for i in range(patches.shape[1]):
            patch = patches[:, i, :, :].unsqueeze(1)
            xloc = self.local_feat(patch)  # local_feat通道数要改成patch的数量
            for n in range(self.nblock):
                att = self.attnlist[n](xloc, xglo)
                tmp = self.fftlist[n](att + xloc)
                xloc = xloc + tmp

            xlocout = self.locout(torch.flatten(self.avg(xloc), 1))
            outlist.append(xlocout)

        return outlist


# if __name__ == '__main__':
#     x1 = torch.rand(5, 1, 256, 256)
#     x2 = torch.rand(5, 2, 40, 40)
# glo_label = torch.Tensor([0, 1, 1, 0, 1]).long()
# loc_label = torch.Tensor([[1, 1, 1, 0, 1], [1, 0, 0, 0, 1]]).long()
#
# mod = GlobalLocalTransformer(inplace=1, nblock=3, backbone='vgg8')
# zlist = mod(x1, x2)
#
# gloout = zlist[0]
# locout = zlist[1:]
#
# lf = LinkCrossEntropy(nn.CrossEntropyLoss())
# loss = lf(gloout, locout, glo_label, loc_label)
# loss.backward()
