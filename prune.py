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


def apply_pruning(model, cfg_mask):
    """
    使用 torch.nn.utils.prune API 在原始模型上应用 Network Slimming 剪枝
    cfg_mask: 每个 BN 层的 mask 列表 (torch.Tensor, shape=[C])
    """
    layer_id_in_cfg = 0
    start_mask = torch.ones(1)  # 输入 RGB 三通道
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            idx1 = torch.nonzero(end_mask).squeeze().tolist()
            if not isinstance(idx1, list):
                idx1 = [idx1]

            # BN 层直接裁剪参数
            m.weight.data = m.weight.data[idx1].clone()
            m.bias.data = m.bias.data[idx1].clone()
            m.running_mean = m.running_mean[idx1].clone()
            m.running_var = m.running_var[idx1].clone()

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m, nn.Conv2d):
            if conv_count == 0:
                # 第一层 Conv 不剪枝
                conv_count += 1
                continue

            idx0 = torch.nonzero(start_mask).squeeze().tolist()
            idx1 = torch.nonzero(end_mask).squeeze().tolist()
            if not isinstance(idx0, list): idx0 = [idx0]
            if not isinstance(idx1, list): idx1 = [idx1]

            # 输入通道 mask
            in_mask = torch.zeros(m.in_channels)
            in_mask[idx0] = 1
            prune.custom_from_mask(m, name="weight", mask=in_mask.view(1, -1, 1, 1).expand_as(m.weight))

            # 输出通道 mask（非残差块最后一层）
            if conv_count % 3 != 1:
                out_mask = torch.zeros(m.out_channels)
                out_mask[idx1] = 1
                prune.custom_from_mask(m, name="weight", mask=out_mask.view(-1, 1, 1, 1).expand_as(m.weight))

            prune.remove(m, "weight")  # 固定剪枝结果
            conv_count += 1

        elif isinstance(m, nn.Linear):
            idx0 = torch.nonzero(start_mask).squeeze().tolist()
            if not isinstance(idx0, list): idx0 = [idx0]

            in_mask = torch.zeros(m.in_features)
            in_mask[idx0] = 1
            prune.custom_from_mask(m, name="weight", mask=in_mask.view(1, -1).expand_as(m.weight))
            prune.remove(m, "weight")

    return model


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
    resNet.to(args.device)
    resNet.eval()
    # acc = test(resNet, args)
    # print(">>> pre-pruned model accuracy: {}".format(acc))
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
    for k, m in enumerate(tqdm(resNet.modules())):
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
    print("pruned_ratio: {:.2f}".format(pruned_ratio))
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
