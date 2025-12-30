import os
import torch
from torchvision import datasets
from lib.model.resNet.resNet import ResNet
from configs.prune import parser
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import torch.nn.utils.prune as prune
import torch.nn as nn


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
    resNet = ResNet()
    resNet.adaptMnist()
    # load checkpoint
    checkpointName = 'resNet_{}_{}_{}.pth'.format(
        args.checkSession, args.checkEpoch, args.checkPoint)
    print(">>> load checkpoint : {}".format(checkpointName))
    checkpointPath = os.path.join(
        args.checkpoint_dir, str(args.checkSession), str(checkpointName))
    resNet.loadCheckpoint(checkpointPath, args.device)
    # to gpu
    resNet.to(args.device)
    resNet.eval()
    # acc = test(resNet, args)
    # print(">>> pre-pruned model accuracy: {}".format(acc))
    # 统计所有 scaling factor(gamma) 的数量
    total = 0
    for m in resNet.modules():
        if isinstance(m, torch.nn.Conv2d):
            total += m.weight.data.shape[0]
    bn = torch.zeros(total)
    index = 0
    for m in resNet.modules():
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
        elif isinstance(m, torch.nn.MaxPool2d):
            cfg.append('M')
    pruned_ratio = pruned/total
    print("pruned_ratio: {:.2f}".format(pruned_ratio))
    print('>>> Pre-processing Successful!')
    # acc = test(resNet, args)
    # print(">>> pruned model accuracy: {}".format(acc))

    # 使用示例
    pruned_model = apply_pruning(resNet, cfg_mask)

    # 保存剪枝后的模型
    torch.save(pruned_model.state_dict(), "pruned_model.pth")

    pass
    

if __name__ == "__main__":
    main()
