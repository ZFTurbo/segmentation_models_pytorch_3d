"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHWD (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ._base import EncoderMixin


pretrained_settings = {
    'dpn68': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-4af7d88d2.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn68b': {
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68b_extra-363ab9c19.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn92': {
        # 'imagenet': {
        #     'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth',
        #     'input_space': 'RGB',
        #     'input_size': [3, 224, 224],
        #     'input_range': [0, 1],
        #     'mean': [124 / 255, 117 / 255, 104 / 255],
        #     'std': [1 / (.0167 * 255)] * 3,
        #     'num_classes': 1000
        # },
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-fda993c95.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn98': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn98-722954780.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn131': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn131-7af84be88.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
    'dpn107': {
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn107_extra-b7f9f4cc9.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    }
}

def dpn68(num_classes=1000, pretrained='imagenet'):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn68'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpn68b(num_classes=1000, pretrained='imagenet+5k'):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn68b'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpn92(num_classes=1000, pretrained='imagenet+5k'):
    model = DPN(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn92'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpn98(num_classes=1000, pretrained='imagenet'):
    model = DPN(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn98'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpn131(num_classes=1000, pretrained='imagenet'):
    model = DPN(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn131'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def dpn107(num_classes=1000, pretrained='imagenet+5k'):
    model = DPN(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
        num_classes=num_classes, test_time_pool=True)
    if pretrained:
        settings = pretrained_settings['dpn107'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm3d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv3d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv3d, self).__init__()
        self.bn = nn.BatchNorm3d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv3d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(
            3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv3d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv3d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv3d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv3d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv3d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv3d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv3d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = OrderedDict()

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)

        self.features = nn.Sequential(blocks)

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.last_linear = nn.Conv3d(in_chs, num_classes, kernel_size=1, bias=True)

    def logits(self, features):
        if not self.training and self.test_time_pool:
            x = F.avg_pool3d(features, kernel_size=7, stride=1)
            out = self.last_linear(x)
            # The extra test time pool should be pooling an img_size//32 - 6 size patch
            out = adaptive_avgmax_pool3d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool3d(features, pool_type='avg')
            out = self.last_linear(x)
        return out.view(out.size(0), -1)

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

""" PyTorch selectable adaptive pooling
Adaptive pooling with the ability to select the type of pooling from:
    * 'avg' - Average pooling
    * 'max' - Max pooling
    * 'avgmax' - Sum of average and max pooling re-scaled by 0.5
    * 'avgmaxc' - Concatenation of average and max pooling along feature dim, doubles feature dim

Both a functional and a nn.Module version of the pooling is provided.

Author: Ross Wightman (rwightman)
"""

def pooling_factor(pool_type='avg'):
    return 2 if pool_type == 'avgmaxc' else 1


def adaptive_avgmax_pool3d(x, pool_type='avg', padding=0, count_include_pad=False):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avgmaxc':
        x = torch.cat([
            F.avg_pool3d(
                x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad),
            F.max_pool3d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        ], dim=1)
    elif pool_type == 'avgmax':
        x_avg = F.avg_pool3d(
                x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
        x_max = F.max_pool3d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        x = 0.5 * (x_avg + x_max)
    elif pool_type == 'max':
        x = F.max_pool3d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
    else:
        if pool_type != 'avg':
            print('Invalid pool type %s specified. Defaulting to average pooling.' % pool_type)
        x = F.avg_pool3d(
            x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
    return x


class AdaptiveAvgMaxPool3d(torch.nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg'):
        super(AdaptiveAvgMaxPool3d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        if pool_type == 'avgmaxc' or pool_type == 'avgmax':
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool3d(output_size), nn.AdaptiveMaxPool3d(output_size)])
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool3d(output_size)
        else:
            if pool_type != 'avg':
                print('Invalid pool type %s specified. Defaulting to average pooling.' % pool_type)
            self.pool = nn.AdaptiveAvgPool3d(output_size)

    def forward(self, x):
        if self.pool_type == 'avgmaxc':
            x = torch.cat([p(x) for p in self.pool], dim=1)
        elif self.pool_type == 'avgmax':
            x = 0.5 * torch.sum(torch.stack([p(x) for p in self.pool]), 0).squeeze(dim=0)
        else:
            x = self.pool(x)
        return x

    def factor(self):
        return pooling_factor(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'output_size=' + str(self.output_size) \
               + ', pool_type=' + self.pool_type + ')'


class DPNEncoder(DPN, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, depth=5, strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)), **kwargs):
        super().__init__(**kwargs)
        self._stage_idxs = stage_idxs
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        self.strides = strides

        del self.last_linear

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.features[0].conv, self.features[0].bn, self.features[0].act),
            nn.Sequential(self.features[0].pool, self.features[1 : self._stage_idxs[0]]),
            self.features[self._stage_idxs[0] : self._stage_idxs[1]],
            self.features[self._stage_idxs[1] : self._stage_idxs[2]],
            self.features[self._stage_idxs[2] : self._stage_idxs[3]],
        ]

    def forward(self, x):

        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            if isinstance(x, (list, tuple)):
                features.append(F.relu(torch.cat(x, dim=1), inplace=True))
            else:
                features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        from segmentation_models_pytorch_3d.utils.convert_weights import convert_2d_weights_to_3d
        state_dict.pop("last_linear.bias", None)
        state_dict.pop("last_linear.weight", None)
        state_dict = convert_2d_weights_to_3d(state_dict)
        super().load_state_dict(state_dict, **kwargs)


dpn_encoders = {
    "dpn68": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn68"],
        "params": {
            "stage_idxs": (4, 8, 20, 24),
            "out_channels": (3, 10, 144, 320, 704, 832),
            "groups": 32,
            "inc_sec": (16, 32, 32, 64),
            "k_r": 128,
            "k_sec": (3, 4, 12, 3),
            "num_classes": 1000,
            "num_init_features": 10,
            "small": True,
            "test_time_pool": True,
        },
    },
    "dpn68b": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn68b"],
        "params": {
            "stage_idxs": (4, 8, 20, 24),
            "out_channels": (3, 10, 144, 320, 704, 832),
            "b": True,
            "groups": 32,
            "inc_sec": (16, 32, 32, 64),
            "k_r": 128,
            "k_sec": (3, 4, 12, 3),
            "num_classes": 1000,
            "num_init_features": 10,
            "small": True,
            "test_time_pool": True,
        },
    },
    "dpn92": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn92"],
        "params": {
            "stage_idxs": (4, 8, 28, 32),
            "out_channels": (3, 64, 336, 704, 1552, 2688),
            "groups": 32,
            "inc_sec": (16, 32, 24, 128),
            "k_r": 96,
            "k_sec": (3, 4, 20, 3),
            "num_classes": 1000,
            "num_init_features": 64,
            "test_time_pool": True,
        },
    },
    "dpn98": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn98"],
        "params": {
            "stage_idxs": (4, 10, 30, 34),
            "out_channels": (3, 96, 336, 768, 1728, 2688),
            "groups": 40,
            "inc_sec": (16, 32, 32, 128),
            "k_r": 160,
            "k_sec": (3, 6, 20, 3),
            "num_classes": 1000,
            "num_init_features": 96,
            "test_time_pool": True,
        },
    },
    "dpn107": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn107"],
        "params": {
            "stage_idxs": (5, 13, 33, 37),
            "out_channels": (3, 128, 376, 1152, 2432, 2688),
            "groups": 50,
            "inc_sec": (20, 64, 64, 128),
            "k_r": 200,
            "k_sec": (4, 8, 20, 3),
            "num_classes": 1000,
            "num_init_features": 128,
            "test_time_pool": True,
        },
    },
    "dpn131": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn131"],
        "params": {
            "stage_idxs": (5, 13, 41, 45),
            "out_channels": (3, 128, 352, 832, 1984, 2688),
            "groups": 40,
            "inc_sec": (16, 32, 32, 128),
            "k_r": 160,
            "k_sec": (4, 8, 28, 3),
            "num_classes": 1000,
            "num_init_features": 128,
            "test_time_pool": True,
        },
    },
}
