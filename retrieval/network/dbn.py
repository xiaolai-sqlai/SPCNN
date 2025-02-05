from __future__ import absolute_import

import copy
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
from functools import reduce

from network.layer import BatchDrop, BatchErasing
from network.spcnn import SPCNN
from network.inceptionnext import inceptionnext_tiny
from network.moganet import moganet_small
from network.unireplknet import unireplknet_t
from torchvision.models import resnet50

class GlobalAvgPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalAvgPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class GlobalMaxPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalMaxPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class DBN(nn.Module):
    def __init__(self, num_classes=751, num_parts=[1,2], feat_num=0, std=0.1, net="regnet_y_1_6gf", drop=0.0, erasing=0.0):
        super(DBN, self).__init__()
        self.num_parts = num_parts
        self.feat_num = feat_num
        if self.training:
            self.batch_erasing = nn.Identity()
            if drop > 0:
                self.batch_erasing = BatchDrop(drop=drop)
            elif erasing > 0:
                self.batch_erasing = BatchErasing(smax=erasing)

        if net == "spcnn_small":
            pool_num = 1024
            base = SPCNN(dims=[64,128,256,512], layers=[4,7,15,4], mlp_ratio=3.0, expand_ratio=1.0, drop_path_rate=0.20)
            path = "pretrain/checkpoint_small_4.4G.pth"
            base.load_state_dict(torch.load(path), "cpu")

            self.stem = base.first_conv
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3

            base.layer4[0].mlp[1].conv.stride = (1, 1)
            base.layer4[0].skip[0].conv.stride = (1, 1)

            self.branch_1 = copy.deepcopy(nn.Sequential(base.layer4, base.head))
            self.branch_2 = copy.deepcopy(nn.Sequential(base.layer4, base.head))
        elif net == "moganet_small":
            pool_num = 512
            base = moganet_small()
            path = "pretrain/moganet_small_sz224_8xbs128_ep300.pth"
            base.load_state_dict(torch.load(path), "cpu")

            self.stem = base.patch_embed1
            layer1 = [b for b in base.blocks1] + [base.norm1]
            self.layer1 = nn.Sequential(*layer1)
            layer2 = [base.patch_embed2] + [b for b in base.blocks2] + [base.norm2]
            self.layer2 = nn.Sequential(*layer2)
            layer3 = [base.patch_embed3] + [b for b in base.blocks3] + [base.norm3]
            self.layer3 = nn.Sequential(*layer3)

            base.patch_embed4.projection.stride = (1, 1)
            layer4 = [base.patch_embed4] + [b for b in base.blocks4] + [base.norm4]
            self.branch_1 = copy.deepcopy(nn.Sequential(*layer4))
            self.branch_2 = copy.deepcopy(nn.Sequential(*layer4))
        elif net == "unireplknet_t":
            pool_num = 640
            base = unireplknet_t()
            path = "pretrain/unireplknet_t_in1k_224_acc83.21.pth"
            base.load_state_dict(torch.load(path), "cpu")

            self.stem = base.downsample_layers[0]
            self.layer1 = base.stages[0]
            self.layer2 = nn.Sequential(base.downsample_layers[1], base.stages[1])
            self.layer3 = nn.Sequential(base.downsample_layers[2], base.stages[2])

            base.downsample_layers[3][0].stride = (1, 1)
            self.branch_1 = copy.deepcopy(nn.Sequential(base.downsample_layers[3], base.stages[3]))
            self.branch_2 = copy.deepcopy(nn.Sequential(base.downsample_layers[3], base.stages[3]))
        elif net == "inceptionnext_tiny":
            pool_num = 768
            base = inceptionnext_tiny()
            path = "pretrain/inceptionnext_tiny.pth"
            base.load_state_dict(torch.load(path), "cpu")

            self.stem = base.stem
            self.layer1 = base.stages[0]
            self.layer2 = base.stages[1]
            self.layer3 = base.stages[2]

            base.stages[3].downsample[1].stride = (1, 1)
            self.branch_1 = copy.deepcopy(base.stages[3])
            self.branch_2 = copy.deepcopy(base.stages[3])
        elif net == "resnet50":
            pool_num = 2048
            base = resnet50(pretrained=True)

            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3

            base.layer4[0].conv2.stride = (1, 1)
            base.layer4[0].downsample[0].stride = (1, 1)
            self.branch_1 = copy.deepcopy(base.layer4)
            self.branch_2 = copy.deepcopy(base.layer4)


        self.pool_list = nn.ModuleList()
        self.feat_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.class_list = nn.ModuleList()

        for i in range(len(self.num_parts)):
            self.pool_list.append(GlobalAvgPool2d(p=1))
            if self.feat_num == 0:
                feat_num = pool_num
                feat = nn.Identity()
            else:
                feat = nn.Linear(pool_num, feat_num, bias=False)
                init.kaiming_normal_(feat.weight)
            self.feat_list.append(feat)
            bn = nn.BatchNorm1d(feat_num)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            self.bn_list.append(bn)

            linear = nn.Linear(feat_num, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

        for i in range(sum(self.num_parts)):
            self.pool_list.append(GlobalMaxPool2d(p=1))
            if self.feat_num == 0:
                feat_num = pool_num
                feat = nn.Identity()
            else:
                feat = nn.Linear(pool_num, feat_num, bias=False)
                init.kaiming_normal_(feat.weight)
            self.feat_list.append(feat)
            bn = nn.BatchNorm1d(feat_num)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            bn.bias.requires_grad = False
            self.bn_list.append(bn)

            linear = nn.Linear(feat_num, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)


    def forward(self, x):
        if self.training:
            x = self.batch_erasing(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        x_chunk = [x1, x2, x1] + list(torch.chunk(x2, dim=2, chunks=self.num_parts[1]))

        pool_list = []
        feat_list = []
        bn_list = []
        class_list = []

        for i in range(sum(self.num_parts)+len(self.num_parts)):
            pool = self.pool_list[i](x_chunk[i]).flatten(1)
            pool_list.append(pool)
            feat = self.feat_list[i](pool)
            feat_list.append(feat)
            bn = self.bn_list[i](feat)
            bn_list.append(bn)
            feat_class = self.class_list[i](bn)
            class_list.append(feat_class)

        if self.training:
            return class_list, bn_list[:2]
        return bn_list

