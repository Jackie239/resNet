import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from configs.prune import parser
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.utils import loadCheckpoint
from lib.model.resNet.resnet_fpn import ResNet, BasicBlockWithSelect, channel_selection
import numpy as np


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
    loss_avg_temp = 0.0
    acc_avg_temp = 0.0
    for step, (images, label) in tqdm(enumerate(DataLoaderTest)):
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
    # resNet.loadCheckpoint(checkpointPath, args.device)
    loadCheckpoint(resnet18, checkpointPath, device="cpu", replace=True, strict=False)

    # to gpu
    resnet18.to(args.device)
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
    
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in tqdm(enumerate(resnet18.modules())):
        if isinstance(m, torch.nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            # tensor.gt(threshold)  大于threshold的位置为True，否则为False
            mask = weight_copy.gt(thre).float().to(args.device)
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # mask=0 的gamma和beta被置0
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        # M 不再加入
        # elif isinstance(m, torch.nn.MaxPool2d):
        #     cfg.append('M')
    pruned_ratio = pruned/total
    print('>>> Pre-processing Successful!')
    # acc = test(resNet, args)
    # print(">>> pruned model accuracy: {}".format(acc))
    print("Cfg: {}".format(cfg))
    # build a pruned model
    resnet18_pruned = ResNet(BasicBlockWithSelect, [2, 2, 2, 2], num_classes=10, cfg=cfg)
    resnet18_pruned.conv1 = nn.Conv2d(
        in_channels=1, out_channels=resnet18_pruned.conv1.out_channels, kernel_size=resnet18_pruned.conv1.kernel_size,
        stride=resnet18_pruned.conv1.stride, padding=resnet18.conv1.padding, bias=resnet18_pruned.conv1.bias)

    old_modules = list(resnet18.modules())
    new_modules = list(resnet18_pruned.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(1)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    bn_count = 0
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            bn_count += 1
            # stem 中的 bn 不剪枝
            if bn_count == 1:
                m1.weight.data = m0.weight.data.clone()
                layer_id_in_cfg += 1
            # block 中，主分支的最后一个 bn 和短接上的最后一个 bn 不剪枝
            # 主分支的最后一个 bn 的上两个应该是 ReLU
            # 短接上的最后一个 bn 的下一个是 select
            elif (isinstance(old_modules[layer_id-2], torch.nn.ReLU)
            or (isinstance(old_modules[layer_id+1], channel_selection))):
                m1.weight.data = m0.weight.data.clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            # 主分支的其余要剪枝
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, channel_selection):
            idx1 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # We need to set the channel selection layer.
            m2 = new_modules[layer_id]
            # [B, C, 1, 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0
        elif isinstance(m0, torch.nn.Conv2d):
            conv_count += 1
            # stem 中的 conv 不剪枝
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
            # 短接上的 conv 不剪枝
            # 短接的上一个应该是 bn
            elif isinstance(old_modules[layer_id - 1], torch.nn.BatchNorm2d):
                m1.weight.data = m0.weight.data.clone()
            # 主分支的第一个 conv 要剪枝
            # 主分支的第一个 conv 的上一个是 select
            elif isinstance(old_modules[layer_id - 1], channel_selection):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
            # 主分支的最后一个 conv 只剪枝通道
            # 主分支的最后一个 conv 上一个是ReLU
            elif isinstance(old_modules[layer_id - 1], torch.nn.ReLU):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
        # fc 不剪枝
    pass

if __name__ == "__main__":
    main()
