from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

from numpy.ma.core import identity

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer.
    It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes`
        which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockWithSelect(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(BasicBlockWithSelect, self).__init__()
        self.select = channel_selection(inplanes)
        # cfg [64, 64, 64] for resnet18
        #     [[previous block bn], [current block bn]]
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(cfg[0], cfg[1], stride)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[1], planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.select(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckWithSelect(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckWithSelect, self).__init__()
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, cfg=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        if cfg is None:
            # resnet 18
            # 每个stage的block数量
            if layers == [2, 2, 2, 2]:
                cfg = [[self.inplanes],
                       [64, 64]*layers[0],
                       [128, 128, 128], [128, 128]*(layers[1]-1),
                       [256, 256, 256], [256, 256]*(layers[2]-1),
                       [512, 512, 512], [512, 512]*(layers[3]-1)]
                self.stageStride = [1, 5, 10, 15]
                self.block1Stride = [1, 3]
                self.block2Stride = [1, 4]
                # 这里只包含主分支的 bn channel，main中cfg需要将短接的bn去掉，或者在这里去掉
                self.block_bn_count = 2
                # cfg = [
                #     [self.inplanes],
                #     [64, 64]*layers[0],
                #     [128, 128]*(layers[1]),
                #     [256, 256]*(layers[2]),
                #     [512, 512]*(layers[3])]
                self.cfg = [item for sub_list in cfg for item in sub_list]

            # resnet 34
            elif layers == [3, 4, 6, 3]:
                pass
            # resnet 50
            elif layers == [3, 4, 6, 3]:
                pass
            # resnet 101
            elif layers == [3, 4, 23, 3]:
                pass
            else:
                raise ValueError("Invalid layers configuration: layers {}".format(layers))
        else:

            if layers == [2, 2, 2, 2]:
                self.block_bn_count = 2
                self.cfg = cfg
                self.stageStride = [1, 5, 10, 15]
                self.block1Stride = [1, 3]
                self.block2Stride = [1, 4]
                # # 去掉 cfg 中短接部分的 bn channel
                # self.block_bn_count = 2
                # idxRemove = []
                # stage_stride = 1
                # for stage, block_num in enumerate(layers):
                #     if stage == 0:
                #         stage_stride += self.block_bn_count*block_num
                #     else:
                #         idxRemove.append(stage_stride+self.block_bn_count)
                #         stage_stride += (self.block_bn_count*block_num+1)
                # idxRemove_set = set(idxRemove)
                # self.cfg = [item for i, item in enumerate(cfg) if i not in idxRemove_set]
            elif layers == [3, 4, 6, 3]:
                pass
            elif layers == [3, 4, 6, 3]:
                pass
            elif layers == [3, 4, 23, 3]:
                pass
            else:
                raise ValueError("Invalid layers configuration: layers {}".format(layers))



        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        # stage 1-4
        self.layer1 = self._make_layer(block, 64, layers[0], self.cfg[self.stageStride[0]-1:self.stageStride[0]+self.block_bn_count*layers[0]], self.block1Stride, isStage1=True)
        self.layer2 = self._make_layer(block, 128, layers[1], self.cfg[self.stageStride[1]-1:self.stageStride[1]+self.block_bn_count*layers[1]+1], self.block2Stride, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], self.cfg[self.stageStride[2]-1:self.stageStride[2]+self.block_bn_count*layers[2]+1], self.block2Stride, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], self.cfg[self.stageStride[3]-1:self.stageStride[3]+self.block_bn_count*layers[3]+1], self.block2Stride, stride=2)
        # it is slightly better whereas slower to set stride = 1
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, blockStride, stride=1, isStage1=False):
        if not isStage1:
            downsmpleStride = 1
        else:
            downsmpleStride = 0
        # [64, 64, 64, 60, 64]
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        # cfg = [64, 64, 64, 64, 64]
        #           [current stage ]
        layers = []
        layers.append(block(self.inplanes, planes, cfg[blockStride[0]-1:blockStride[0]+self.block_bn_count+downsmpleStride], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[blockStride[i]-1:blockStride[i]+self.block_bn_count+1]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



