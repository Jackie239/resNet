import torch
import numpy as np
from lib.model.resNet.resnet_fpn import channel_selection



def pruneBN(layer_id, moduleName, m0, m1, idx1=None):
    # print(">>> layer_id: {}, moduleName: {}".format(layer_id, moduleName))
    if idx1 is None:
        # print("prune state :{}".format(not(idx1 is None)))
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    else:
        # print("prune state :{}".format(not(idx1 is None)))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        # print("old module :{}".format(m0))
        # print("old module weight size:{}".format(m0.weight.data.size()))
        # print("new module :{}".format(m1))
        # print("new module weight size:{}".format(m1.weight.data.size()))


def symmetricDifference(mask_main, mask_identity):
    # 找同为0的位置并取反
    mask_result = (~((mask_main == 0) & (mask_identity == 0))).float()
    return mask_result


def bitOr(mask1, mask2):
    # return mask1 | mask2
    return torch.logical_or(mask1.bool(), mask2.bool()).to(torch.float)


def getBNmask(bn, thre):
    weight_copy = bn.weight.data.abs().clone()
    mask = weight_copy.gt(thre).float().to(bn.weight.device)
    return mask


def traverse_refine(module, cfg_dict_refine, blockName, depth=0,
                    skipStem=True):
    for name, child in module.named_children():
        # 跳过最开始的 stem 部分
        if skipStem and name in ["conv1", "bn1", "relu", "maxpool"]:
            if name == "bn1":
                mask = getBNmask(child, cfg_dict_refine["thre"])
                cfg_dict_refine["prunedCount"] += mask.shape[0] - torch.sum(mask)
                # mask=0 的gamma和beta被置0
                child.weight.data.mul_(mask)
                child.bias.data.mul_(mask)
                cfg_dict_refine["cfg"].append(int(torch.sum(mask)))
                cfg_dict_refine["cfg_mask"].append(mask.clone())
                cfg_dict_refine["stemBN_mask"] = mask.clone()
            continue
        # 处理 BatchNorm2d
        elif isinstance(child, torch.nn.BatchNorm2d):
            mask = getBNmask(child, cfg_dict_refine["thre"])
            cfg_dict_refine["prunedCount"] += mask.shape[0] - torch.sum(mask)
            # mask=0 的gamma和beta被置0
            child.weight.data.mul_(mask)
            child.bias.data.mul_(mask)
            cfg_dict_refine["cfg"].append(int(torch.sum(mask)))
            cfg_dict_refine["cfg_mask"].append(mask.clone())

            if name == "bn2" and blockName == "BasicBlockWithSelect":           # 主分支最后 BN
                cfg_dict_refine["currentBlock"]["mainLastBN_mask"] = mask.clone()
            elif name == "bn3" and blockName == "BottleneckWithSelect":
                cfg_dict_refine["currentBlock"]["mainLastBN_mask"] = mask.clone()
            elif name == "1":           # identity 分支最后 BN (downsample里的BN)
                cfg_dict_refine["currentBlock"]["idenLastBN_mask"] = mask.clone()

            # 对 bn 计数，与 cfg 索引一致
            cfg_dict_refine["bnCount"] += 1
        # 处理 select 层
        elif name == "select":
            cfg_dict_refine["currentBlock"]["select_mask"] = child.indexes
            if cfg_dict_refine["stageIdx"] == 0:
                # 第一个 block 的 select
                if cfg_dict_refine["blockIdx"] == 0:
                    mask_res = cfg_dict_refine["stemBN_mask"]
                    cfg_dict_refine["select_mask_list"].append(mask_res)
                # 其余的 block 的 select
                else:
                    # res18的第二个block前是不带downsample的block
                    if blockName == "BasicBlockWithSelect" and cfg_dict_refine["blockIdx"] == 1:
                        mask_res = bitOr(
                            cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                            cfg_dict_refine["previousBlock"]["select_mask"])
                        cfg_dict_refine["select_mask_list"].append(mask_res)
                    # res50的第二个block前是带downsample的block
                    elif blockName == "BottleneckWithSelect" and cfg_dict_refine["blockIdx"] == 1:
                        mask_res = bitOr(
                            cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                            cfg_dict_refine["previousBlock"]["idenLastBN_mask"])
                        cfg_dict_refine["select_mask_list"].append(mask_res)
                    # 剩下的block前都是不带downsample的block
                    else:
                        mask_res = bitOr(
                            cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                            cfg_dict_refine["previousBlock"]["select_mask"])
                        cfg_dict_refine["select_mask_list"].append(mask_res)
            elif cfg_dict_refine["stageIdx"] in [1, 2, 3]:
                # stage 2-4 的第一个block的前一个不含downsample的block
                if cfg_dict_refine["blockIdx"] == 0:
                    mask_res = bitOr(
                        cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                        cfg_dict_refine["previousBlock"]["select_mask"])
                    cfg_dict_refine["select_mask_list"].append(mask_res)
                # 第二个block的前一个是包含downsample的block
                elif cfg_dict_refine["blockIdx"] == 1:
                    mask_res = bitOr(
                        cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                        cfg_dict_refine["previousBlock"]["idenLastBN_mask"])
                    cfg_dict_refine["select_mask_list"].append(mask_res)
                # 其余的block前都是不包含downsample的block
                else:
                    mask_res = bitOr(
                        cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                        cfg_dict_refine["previousBlock"]["select_mask"])
                    cfg_dict_refine["select_mask_list"].append(mask_res)
            cfg_dict_refine["select_cfg"].append(mask_res.sum())
        # 如果遇到一个 block
        elif child.__class__.__name__ == blockName:
            traverse_refine(child, cfg_dict_refine, blockName, depth + 1, skipStem=False)

            # block结束后更新
            cfg_dict_refine["previousBlock"] = cfg_dict_refine["currentBlock"].copy()
            cfg_dict_refine["currentBlock"] = {"mainLastBN_mask": None, "idenLastBN_mask": None, "select_mask": None}

            cfg_dict_refine["blockIdx"] += 1
            continue
        # 如果遇到一个 stage (layer1, layer2, layer3, layer4)
        elif name in ["layer1", "layer2", "layer3", "layer4"]:
            traverse_refine(child, cfg_dict_refine, blockName, depth + 1, skipStem=False)

            # stage结束后更新
            cfg_dict_refine["stageIdx"] += 1
            cfg_dict_refine["blockIdx"] = 0  # 重置block计数
            continue

        # 递归进入其他子模块
        traverse_refine(child, cfg_dict_refine, blockName, depth + 1, skipStem=False)


def gammathreshold(resNet, slimmingPercentage):

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
    thre_index = int(total * slimmingPercentage)
    return y[thre_index]


def pruneModle(softModel, hardModel, cfg_dict, blockType, layers):
    cfg_mask = cfg_dict["cfg_mask"]
    cfg_mask_select = cfg_dict["select_mask_list"]
    old_modules = list(softModel.modules())
    new_modules = list(hardModel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(1)
    end_mask = cfg_mask[layer_id_in_cfg]
    selectIdx = 0
    bnIdxInBlock = 0    # range [0, 1]
    stageIdx = 0        # range [0, 1, 2, 3]
    blockIdxInStage = 0
    if blockType == "BasicBlockWithSelect":
        mainLastBNIdxInBlock = 1
    else:
        mainLastBNIdxInBlock = 2
    moduleNameList = ["Conv2d", "BatchNorm2d", "channel_selection", "Linear"]
    moduleIdxList = []
    for layer_id in range(len(old_modules)):
        m1 = new_modules[layer_id]
        if m1.__class__.__name__ in moduleNameList:
            moduleIdxList.append(layer_id)
    for i, idx in enumerate(moduleIdxList):
        print("stage: {}, block: {}, bn: {}".format(stageIdx, blockIdxInStage, bnIdxInBlock))
        m0 = old_modules[idx]
        m1 = new_modules[idx]
        # stem conv
        if i == 0 and isinstance(m0, torch.nn.Conv2d):
            m1.weight.data = m0.weight.data.clone()
            # bn
        elif i == 1 and isinstance(m0, torch.nn.BatchNorm2d):
            # stem 中的bn不剪枝
            pruneBN(idx, m0.__class__.__name__, m0, m1)
            # update cfg_mask
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        # stage 1-4
        else:
            if isinstance(m0, channel_selection):
                idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_select[selectIdx].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                # We need to set the channel selection layer.
                # [B, C, 1, 1]
                m1.indexes.data.zero_()
                m1.indexes.data[idx0.tolist()] = 1.0
            elif isinstance(m0, torch.nn.Conv2d):
                # 主分支第一个conv剪枝B和C，且C来自select
                if bnIdxInBlock == 0 :
                    idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_select[selectIdx].cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    m1.weight.data = w1.clone()
                # 主分支最后一个conv只剪枝C
                elif bnIdxInBlock == mainLastBNIdxInBlock:
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                    m1.weight.data = w1.clone()
                # 短接的conv只剪枝C，C来自select
                elif bnIdxInBlock == (mainLastBNIdxInBlock+1):
                    idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask_select[selectIdx].cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                    m1.weight.data = w1.clone()
                # 其余的剪枝B和C，且C来自BN
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
            elif isinstance(m0, torch.nn.BatchNorm2d):
                # 当前 bn 对应的 mask
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                # 主分支最后一个bn和短接的bn不剪枝
                if bnIdxInBlock == (mainLastBNIdxInBlock) or bnIdxInBlock == (mainLastBNIdxInBlock+1):
                    pruneBN(idx, m0.__class__.__name__, m0, m1)
                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(cfg_mask):
                        end_mask = cfg_mask[layer_id_in_cfg]
                # 其余的要剪枝
                else:
                    pruneBN(idx, m0.__class__.__name__, m0, m1, idx1=idx1)
                    layer_id_in_cfg += 1
                    start_mask = end_mask.clone()
                    if layer_id_in_cfg < len(cfg_mask):
                        end_mask = cfg_mask[layer_id_in_cfg]
                # update stageIdx, blockIdx, bnIdxInBlock
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
                    # 更新select
                    selectIdx += 1
                    # 如果当前 stage 的 block 也走完了，进入下一 stage
                    if blockIdxInStage == layers[stageIdx]:
                        stageIdx += 1
                        blockIdxInStage = 0
                else:
                    bnIdxInBlock += 1
            elif isinstance(m0, torch.nn.Linear):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()