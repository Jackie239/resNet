import sys

sys.path.append("d:\WorkSpace\mnist_ResNet")
import torch.nn as nn
import torch
import pickle

from configs.prune import parser
from lib.utils.common import loadCheckpoint
from lib.model.resNet.resnet_fpn import ResNet, BasicBlockWithSelect
from lib.utils.prune import traverse_refine


args = parser.parse_args([])
# build non-pruned model
resnet18 = ResNet(BasicBlockWithSelect, [2, 2, 2, 2], num_classes=10)
resnet18.conv1 = nn.Conv2d(
    in_channels=1, out_channels=resnet18.conv1.out_channels, kernel_size=resnet18.conv1.kernel_size,
    stride=resnet18.conv1.stride, padding=resnet18.conv1.padding, bias=resnet18.conv1.bias)
checkpointPath = "../output/checkpoints/2/resNet_2_10_469.pth"
loadCheckpoint(resnet18, checkpointPath, device="cpu", replace=True, strict=False)
resnet18.to(args.device)
resnet18.eval()
# soft prune
total = 0
for m in resnet18.modules():
    if isinstance(m, torch.nn.Conv2d):
        total += m.weight.data.shape[0]
bn = torch.zeros(total)
index = 0
for m in resnet18.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size
y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

cfg_dict_refine = {
    "stemBN_mask": None,
    "previousBlock": {"mainLastBN_mask": None, "idenLastBN_mask": None, "select_mask": None},
    "currentBlock": {"mainLastBN_mask": None, "idenLastBN_mask": None, "select_mask": None},
    "stageIdx": 0,
    "blockIdx": 0,
    "select_mask_list":[],
    "bnCount":0,
    "select_cfg":[],
    "prunedCount":0,
    "cfg": [],
    "cfg_mask": [],
    "thre": thre
}

traverse_refine(resnet18, args, cfg_dict_refine)

"""
用新的 select_mask_list 的 size，替换channel_select_layer 后面的层的cfg，
然后在构造的时候使用新的 select_mask_list,
"""
dict_path = "./cache/cfg_dict.pkl"
with open (dict_path, "wb") as f:
    pickle.dump(cfg_dict_refine, f)

pass
