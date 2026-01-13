import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.utils.common import loadCheckpoint
from configs.prune import parser
from lib.model.resNet.resnet_fpn import ResNet, BasicBlockWithSelect, channel_selection
from lib.utils.prune import pruneBN, traverse_refine


def test(model, args):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))])
    mnist_test = datasets.FashionMNIST(
        root=args.dataPath, train=False, transform=transform, download=False)
    testSize = len(mnist_test)
    DataLoaderTest = DataLoader(
        mnist_test, batch_size=args.batchSize, num_workers=args.num_workers)

    iters_per_epoch = int(((testSize + args.batchSize - 1) / args.batchSize))
    acc_avg_temp = 0.0
    for step, (images, label) in enumerate(tqdm(DataLoaderTest)):
        # move data to device
        images = images.to(args.device)
        label = label.to(args.device)
        # forward
        scores = model(images)
        label_hat = scores.argmax(dim=1)

        with torch.no_grad():
            acc_avg_temp += (label_hat == label).float().mean().item()
    # compute loss and accuracy average
    acc_avg_temp /= iters_per_epoch
    return acc_avg_temp


def main():
    args = parser.parse_args()
    print(args)
    # >>> build soft-pruned model
    resnet18 = ResNet(BasicBlockWithSelect, [2, 2, 2, 2], num_classes=10)
    resnet18.conv1 = nn.Conv2d(
        in_channels=1, out_channels=resnet18.conv1.out_channels, kernel_size=resnet18.conv1.kernel_size,
        stride=resnet18.conv1.stride, padding=resnet18.conv1.padding, bias=resnet18.conv1.bias)
    # load checkpoint
    checkpointName = 'resNet_{}_{}_{}.pth'.format(
        args.checkSession, args.checkEpoch, args.checkPoint)
    print(">>> load checkpoint : {}".format(checkpointName))
    checkpointPath = os.path.join(
        args.checkpoint_dir, str(args.checkSession), str(checkpointName))
    loadCheckpoint(resnet18, checkpointPath, device="cpu", replace=True, strict=False)
    # to gpu
    resnet18.to(args.device)
    resnet18.eval()
    # acc = test(resnet18, args)
    # print(">>> original model accuracy: {}".format(acc))
    # pruned threshold of gamma
    # 统计所有 scaling factor(gamma) 的数量
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
    # pruned = 0
    # cfg = []
    # cfg_mask = []
    # for k, m in enumerate(tqdm(resnet18.modules())):
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         weight_copy = m.weight.data.abs().clone()
    #         # tensor.gt(threshold)  大于threshold的位置为True，否则为False
    #         mask = weight_copy.gt(thre).float().to(args.device)
    #         pruned = pruned + mask.shape[0] - torch.sum(mask)
    #         # mask=0 的gamma和beta被置0
    #         m.weight.data.mul_(mask)
    #         m.bias.data.mul_(mask)
    #         cfg.append(int(torch.sum(mask)))
    #         cfg_mask.append(mask.clone())
    #         print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
    #             format(k, mask.shape[0], int(torch.sum(mask))))
    #     # M 不再加入
    #     # elif isinstance(m, torch.nn.MaxPool2d):
    #     #     cfg.append('M')
    # print('>>> Pre-processing Successful!')
    # pruned_ratio = pruned/total
    # print("pruned_ratio: {:.2f}".format(pruned_ratio))
    # # acc = test(resnet18, args)
    # # print("soft pruned model accuracy: {}".format(acc))
    # print("Cfg: {}".format(cfg)
    # generate build cfg of pruning and do soft pruning
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
    traverse_refine(resnet18, args, cfg_dict)
    # acc = test(resnet18, args)
    # print("soft pruned model accuracy: {}".format(acc))

    # >>> build pruned model
    cfg = cfg_dict["cfg"]
    cfg_select = cfg_dict["select_cfg"]
    cfg_mask = cfg_dict["cfg_mask"]
    cfg_mask_select = cfg_dict["select_mask_list"]
    cfg_select = [int(cfg) for cfg in cfg_select]
    resnet18_pruned = ResNet(BasicBlockWithSelect, [2, 2, 2, 2], num_classes=10, cfg=cfg, cfg_select=cfg_select)
    resnet18_pruned.conv1 = nn.Conv2d(
        in_channels=1, out_channels=resnet18_pruned.conv1.out_channels, kernel_size=resnet18_pruned.conv1.kernel_size,
        stride=resnet18_pruned.conv1.stride, padding=resnet18_pruned.conv1.padding, bias=resnet18_pruned.conv1.bias)

    old_modules = list(resnet18.modules())
    new_modules = list(resnet18_pruned.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(1)
    end_mask = cfg_mask[layer_id_in_cfg]
    selectIdx = -1
    conv_count = 0
    bn_count = 0
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            # 当前 bn 对应的 mask
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            bn_count += 1
            # stem 中的 bn 不剪枝
            if bn_count == 1:
                pruneBN(layer_id, m0.__class__.__name__, m0, m1)
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            # block 中，主分支的最后一个 bn 和短接上的最后一个 bn 不剪枝
            # 主分支的最后一个 bn 的上两个应该是 ReLU
            # 短接上的最后一个 bn 的下一个是 select
            elif (isinstance(old_modules[layer_id-2], torch.nn.ReLU)
            or (isinstance(old_modules[layer_id+2], channel_selection))):
                pruneBN(layer_id, m0.__class__.__name__, m0, m1)
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            # 主分支的其余要剪枝
            else:
                pruneBN(layer_id, m0.__class__.__name__, m0, m1, idx1=idx1)
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, channel_selection):
            # idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_select[selectIdx].cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            # We need to set the channel selection layer.
            # [B, C, 1, 1]
            m1.indexes.data.zero_()
            m1.indexes.data[idx0.tolist()] = 1.0
        elif isinstance(m0, torch.nn.Conv2d):
            conv_count += 1
            # stem 中的 conv 不剪枝
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
            # 主分支的第一个 conv 剪枝 C和B, 上一个是 select
            elif isinstance(old_modules[layer_id - 1], channel_selection):
                idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_select[selectIdx].cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
            # 短接的 conv 剪枝 C, 短接的上一个应该是 Sequential
            elif isinstance(old_modules[layer_id - 1], Sequential):
                idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_select[selectIdx].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
            # 主分支的最后一个 conv 只剪枝 C, 上一个是ReLU
            elif isinstance(old_modules[layer_id - 1], torch.nn.ReLU):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
            # 其余剪枝 B 和 C
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        # fc 不剪枝
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
        elif isinstance(m0, BasicBlockWithSelect):
            # 遇见block更新
            selectIdx += 1
        # else:
            # print("{} dont prune".format(m0.__class__.__name__))

    print(">>> Successfully build pruned model!")
    # test pruned model
    resnet18_pruned.to(args.device)
    resnet18_pruned.eval()
    # acc = test(resnet18_pruned, args)
    # print(">>> hard pruned model accuracy: {}".format(acc))
    torch.manual_seed(42)
    input = torch.randn(1, 1, 256, 256)
    output_soft = resnet18(input)
    output_hard = resnet18_pruned(input)
    pass
if __name__ == "__main__":
    main()
