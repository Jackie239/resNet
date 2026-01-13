from lib.model.resNet.resnet_fpn import ResNet, BasicBlockWithSelect
from lib.utils.common import loadCheckpoint, test
import torch.nn as nn
import pickle
from configs.prune import parser


def main():
    args = parser.parse_args()
    args.dataPath = "../data"
    print(args)

    # load cfg
    with open("./cache/cfg_dict.pkl", 'rb') as f:
        cfg_dict = pickle.load(f)
    cfg = cfg_dict["cfg"]
    cfg_select = cfg_dict["select_cfg"]
    cfg_select = [int(cfg) for cfg in cfg_select]

    # pre-pruned model
    resnet18 = ResNet(BasicBlockWithSelect, [2, 2, 2, 2], num_classes=10, cfg=cfg, cfg_select=cfg_select)
    # non-pruned model
    # resnet18 = ResNet(BasicBlockWithSelect, [2, 2, 2, 2], num_classes=10)
    resnet18.conv1 = nn.Conv2d(
        in_channels=1, out_channels=resnet18.conv1.out_channels, kernel_size=resnet18.conv1.kernel_size,
        stride=resnet18.conv1.stride, padding=resnet18.conv1.padding, bias=resnet18.conv1.bias)
    checkpointPath = "../output/checkpoints/2/resNet_2_10_469.pth"
    # loadCheckpoint(resnet18, checkpointPath, device="cpu", replace=True, strict=False)
    acc = test(resnet18, args, loadPercent=1.0)
    print("acc: {}".format(acc))
    pass
if __name__ == "__main__":
    main()