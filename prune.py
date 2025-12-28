import os
import torch
from torchvision import datasets
from lib.model.resNet.resNet import ResNet
from configs.prune import parser
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms


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
    for k, m in tqdm(enumerate(resNet.modules())):
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
    print('>>> Pre-processing Successful!')
    acc = test(resNet, args)
    print(">>> pruned model accuracy: {}".format(acc))
    pass
    

if __name__ == "__main__":
    main()
