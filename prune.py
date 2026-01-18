import torch
import torch.nn as nn
from configs.prune import parser
from lib.model.resNet.resnet_fpn import ResNet, BottleneckWithSelect, BasicBlockWithSelect
from lib.utils.prune import traverse_refine, gammathreshold, pruneModle


def main():
    args = parser.parse_args()
    print(args)
    torch.manual_seed(42)
    layers = [3, 4, 6, 3]
    # layers = [2, 2, 2, 2]
    # blockType = "BasicBlockWithSelect"
    blockType = "BottleneckWithSelect"
    # >>> load original model
    resnet = ResNet(eval(blockType), layers, num_classes=10)
    resnet.conv1 = nn.Conv2d(
        in_channels=1, out_channels=resnet.conv1.out_channels, kernel_size=resnet.conv1.kernel_size,
        stride=resnet.conv1.stride, padding=resnet.conv1.padding, bias=resnet.conv1.bias)
    # to gpu
    resnet.to(args.device)
    resnet.eval()
    # get gamma threshold
    thre = gammathreshold(resnet, args.percent)

    cfg_dict = {
        "stemBN_mask": None,
        "previousBlock": {"mainLastBN_mask": None, "idenLastBN_mask": None, "select_mask": None},
        "currentBlock": {"mainLastBN_mask": None, "idenLastBN_mask": None, "select_mask": None},
        "stageIdx": 0,
        "blockIdx": 0,
        "select_mask_list": [],
        "bnCount": 0,
        "select_cfg": [],
        "prunedCount": 0,
        "cfg": [],
        "cfg_mask": [],
        "thre": thre
    }
    # build soft-pruned model
    traverse_refine(resnet, cfg_dict, blockType)

    # >>> build pruned model
    cfg = cfg_dict["cfg"]
    cfg_select = cfg_dict["select_cfg"]

    cfg_select = [int(cfg) for cfg in cfg_select]
    resnet_pruned = ResNet(eval(blockType), layers, num_classes=10, cfg=cfg, cfg_select=cfg_select)
    resnet_pruned.conv1 = nn.Conv2d(
        in_channels=1, out_channels=resnet_pruned.conv1.out_channels, kernel_size=resnet_pruned.conv1.kernel_size,
        stride=resnet_pruned.conv1.stride, padding=resnet_pruned.conv1.padding, bias=resnet_pruned.conv1.bias)

    pruneModle(resnet, resnet_pruned, cfg_dict, blockType, layers)
    print("-----------------------------------")
    print("cfg:")
    print(cfg_dict["cfg"])
    print("cfg_select: ")
    print(cfg_dict["select_cfg"])
    print("-----------------------------------")
    print(">>> Successfully build pruned model!")
    # test pruned model
    resnet_pruned.to(args.device)
    resnet_pruned.eval()
    torch.manual_seed(42)
    input = torch.randn(1, 1, 256, 256)
    output_soft = resnet(input)
    output_hard = resnet_pruned(input)
    error = torch.norm(output_soft - output_hard)
    print("error: {}".format(error))
    pass
if __name__ == "__main__":
    main()
