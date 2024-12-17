import warnings

import torch
import torch.nn as nn
import math

from mmengine.model import BaseModule
from mmseg.registry import MODELS


class DropPath(BaseModule):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
class Act(BaseModule):
    def __init__(self, out_planes=None, act_type="gelu", inplace=True):
        super(Act, self).__init__()

        self.act = None
        if act_type == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == "prelu":
            self.act = nn.PReLU(out_planes)
        elif act_type == "hardswish":
            self.act = nn.Hardswish(inplace=True)
        elif act_type == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act_type == "gelu":
            self.act = nn.GELU()

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        return x


class ConvX(BaseModule):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, act_type="gelu"):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        # self.norm = nn.BatchNorm2d(out_planes)
        self.norm = nn.SyncBatchNorm(out_planes)
        self.act = None
        if act_type is not None:
            self.act = Act(out_planes, act_type) 

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out
    
class LayerNorm(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # with torch.autocast(device_type="cuda", enabled=False):
        #     x = x.float()
        #     var, mean = torch.var_mean(x, dim=self.dim, keepdim=True)
        #     out = (x - mean) / torch.sqrt(var + 1e-6)
        var, mean = torch.var_mean(x, dim=self.dim, keepdim=True)
        out = (x - mean) / torch.sqrt(var + 1e-6)
        return out


class PyramidConvX(BaseModule):
    def __init__(self, planes, act_type="relu"):
        super(PyramidConvX, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            ConvX(planes, planes, groups=planes, kernel_size=3, stride=1, act_type=None)
        )
        self.branch_2 = nn.Sequential(
            nn.AvgPool2d(5, 2, 2),
            ConvX(planes, planes, groups=planes, kernel_size=3, stride=1, act_type=None)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(7, 3, 3),
            ConvX(planes, planes, groups=planes, kernel_size=3, stride=1, act_type=None)
        )
        self.act = Act(planes, act_type)

    def forward(self, x):
        b, c, h, w = x.shape

        x1 = self.branch_1(x)
        x2 = nn.functional.interpolate(self.branch_2(x), size=(h, w), scale_factor=None, mode='nearest')
        x3 = nn.functional.interpolate(self.branch_3(x), size=(h, w), scale_factor=None, mode='nearest')
        out = self.act(x1 + x2 + x3)

        return out
    
class BottleNeck(BaseModule):
    def __init__(self, in_planes, out_planes, stride, act_type="gelu", mlp_ratio=1.0, expand_ratio=1.0, drop_path=0.0):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mid_planes = int(out_planes*mlp_ratio)
        expand_planes = int(out_planes*expand_ratio)

        self.ln = LayerNorm(dim=1)
        self.sblock_in = ConvX(in_planes, mid_planes*2, groups=1, kernel_size=1, stride=1, act_type=act_type)
        self.sblock_dw = ConvX(mid_planes, mid_planes, groups=mid_planes, kernel_size=3, stride=stride, act_type=None)
        self.sblock_proj = ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)

        self.mblock = nn.Sequential(
            ConvX(out_planes, expand_planes, groups=1, kernel_size=1, stride=1, act_type=act_type),
            PyramidConvX(expand_planes, act_type=act_type),
            ConvX(expand_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)
        )

    def forward(self, x):
        out = self.ln(x)
        out = self.sblock_in(out)
        out_1, out_2 = torch.chunk(out, dim=1, chunks=2)
        out = self.sblock_dw(out_1) * out_2
        x = self.drop_path(self.sblock_proj(out)) + x

        x = self.drop_path(self.mblock(x)) + x
        return x
    
class DownBlock(BaseModule):
    def __init__(self, in_planes, out_planes, act_type="gelu", mlp_ratio=1.0, drop_path=0.0):
        super(DownBlock, self).__init__()
        mid_planes = int(out_planes*mlp_ratio)

        self.mlp = nn.Sequential(
            ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, act_type=act_type),
            ConvX(mid_planes, mid_planes, groups=mid_planes, kernel_size=3, stride=2, act_type=act_type),
            ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)
        )

        self.skip = nn.Sequential(
            ConvX(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=2, act_type=None),
            ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.mlp(x)) + self.skip(x)
        return x

@MODELS.register_module()
class SPCNN(BaseModule):
    # pylint: disable=unused-variable
    def __init__(self, dims, layers, act_type="gelu", mlp_ratio=1.0, expand_ratio=1.0, drop_path_rate=0., 
                 out_indices=[0,1,2,3],
                 pretrained=None,
                 init_cfg=None):
        super(SPCNN, self).__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        
        self.out_indices = out_indices
        self.act_type = act_type
        self.mlp_ratio = mlp_ratio
        self.expand_ratio = expand_ratio

        if isinstance(dims, int):
            dims = [dims//2, dims, dims*2, dims*4, dims*8]
        else:
            dims = [dims[0]//2] + dims

        self.first_conv = ConvX(3, dims[0], 1, 3, 2, act_type=act_type)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        self.layer1 = self._make_layers(dims[0], dims[1], layers[0], drop_path=dpr[:layers[0]])
        self.layer2 = self._make_layers(dims[1], dims[2], layers[1], drop_path=dpr[layers[0]:sum(layers[:2])])
        self.layer3 = self._make_layers(dims[2], dims[3], layers[2], drop_path=dpr[sum(layers[:2]):sum(layers[:3])])
        self.layer4 = self._make_layers(dims[3], dims[4], layers[3], drop_path=dpr[sum(layers[:3]):sum(layers[:4])])

    def _make_layers(self, inputs, outputs, num_block, drop_path):
        layers = [DownBlock(inputs, outputs, self.act_type, self.mlp_ratio, drop_path[0])]

        for i in range(1, num_block):
            layers.append(BottleNeck(outputs, outputs, 1, self.act_type, self.mlp_ratio, self.expand_ratio, drop_path[i]))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.first_conv(x)
        x4 = self.layer1(x)
        x8 = self.layer2(x4)
        x16 = self.layer3(x8)
        x32 = self.layer4(x16)

        outs = []
        for i, x in enumerate([x4, x8, x16, x32]):
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
