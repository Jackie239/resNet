from lib.model.resNet.resnet_fpn import ResNet, BasicBlockWithSelect
from lib.utils import loadCheckpoint
import torch.nn as nn


def main():
    cfg = [64, 64, 64, 60, 64, 75,  116, 116, 99,  113, 111, 211, 198, 124, 147, 134, 349, 147, 187, 436]
    resnet18 = ResNet(BasicBlockWithSelect, [2, 2, 2, 2], num_classes=10, cfg=cfg)
    # resnet18 = ResNet(BasicBlockWithSelect, [2, 2, 2, 2], num_classes=10)
    resnet18.conv1 = nn.Conv2d(
        in_channels=1, out_channels=resnet18.conv1.out_channels, kernel_size=resnet18.conv1.kernel_size,
        stride=resnet18.conv1.stride, padding=resnet18.conv1.padding, bias=resnet18.conv1.bias)
    checkpointPath = "../output/checkpoints/2/resNet_2_10_469.pth"
    loadCheckpoint(resnet18, checkpointPath, device="cpu", replace=True, strict=False)
    pass
if __name__ == "__main__":
    main()