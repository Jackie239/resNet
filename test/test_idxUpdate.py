from lib.model.resNet.resnet_fpn import ResNet, BasicBlockWithSelect, channel_selection
import torch
layers = [2, 2, 2, 2]
resnet18 = ResNet(BasicBlockWithSelect, layers, num_classes=10)
resnet18.conv1 = torch.nn.Conv2d(
    in_channels=1, out_channels=resnet18.conv1.out_channels, kernel_size=resnet18.conv1.kernel_size,
    stride=resnet18.conv1.stride, padding=resnet18.conv1.padding, bias=resnet18.conv1.bias)
bnIdxInBlock = 0  # range [0, 1]
stageIdx = 0  # range [0, 1, 2, 3]
blockIdxInStage = 0
bn_count = 0
blockType = resnet18.layer1[0].__class__.__name__
if blockType == "BasicBlockWithSelect":
    mainLastBNIdxInBlock = 1
    idenBNIdxInBlock = 2
else:
    mainLastBNIdxInBlock = 2
    idenBNIdxInBlock = 3
moduleNameList = ["Conv2d", "BatchNorm2d", "channel_selection"]
moduleIdxList = []
old_modules = list(resnet18.modules())
for layer_id in range(len(old_modules)):
    m1 = old_modules[layer_id]
    if m1.__class__.__name__ in moduleNameList:
        moduleIdxList.append(layer_id)
for i, idx in enumerate(moduleIdxList):
    m0 = old_modules[idx]
    if i <= 1:
        continue
    elif isinstance(m0, torch.nn.BatchNorm2d):
        print("stage: {}, block: {}, bn: {}".format(stageIdx, blockIdxInStage, bnIdxInBlock))
        # 判断当前 block 的最后一个 BN
        if stageIdx == 0:
            if blockIdxInStage == 0 and blockType == "BasicBlockWithSelect":
                cur_bn_per_block = mainLastBNIdxInBlock + 1
            elif blockIdxInStage == 0 and blockType == "BottleneckWithSelect":
                cur_bn_per_block = mainLastBNIdxInBlock + 2
            else:
                cur_bn_per_block = mainLastBNIdxInBlock + 1
        else:
            if blockIdxInStage == 0:
                cur_bn_per_block = mainLastBNIdxInBlock + 2
            else:
                cur_bn_per_block = mainLastBNIdxInBlock + 1

        # 更新索引
        if bnIdxInBlock == cur_bn_per_block - 1:  # 当前 block 的最后一个 BN
            blockIdxInStage += 1
            bnIdxInBlock = 0
            # 如果当前 stage 的 block 也走完了，进入下一 stage
            if blockIdxInStage == layers[stageIdx]:
                stageIdx += 1
                blockIdxInStage = 0
        else:
            bnIdxInBlock += 1

