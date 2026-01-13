import torch


def pruneBN(layer_id, moduleName, m0, m1, idx1=None):
    print(">>> layer_id: {}, moduleName: {}".format(layer_id, moduleName))
    if idx1 is None:
        print("prune state :{}".format(not(idx1 is None)))
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    else:
        print("prune state :{}".format(not(idx1 is None)))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        print("old module :{}".format(m0))
        print("old module weight size:{}".format(m0.weight.data.size()))
        print("new module :{}".format(m1))
        print("new module weight size:{}".format(m1.weight.data.size()))


def symmetricDifference(mask_main, mask_identity):
    # 找同为0的位置并取反
    mask_result = (~((mask_main == 0) & (mask_identity == 0))).float()
    return mask_result


def bitOr(mask1, mask2):
    # return mask1 | mask2
    return torch.logical_or(mask1.bool(), mask2.bool()).to(torch.float)


# def writeSelectMask(mask1, mask2, cfg_dict):
#     cfg_dict["select_mask_list"].append(bitOr(mask1, mask2))

def traverse_refine(module, args, cfg_dict_refine, depth=0, skipStem=True):
    for name, child in module.named_children():
        # 跳过最开始的 stem 部分
        if skipStem and name in ["conv1", "bn1", "relu", "maxpool"]:
            if name == "bn1":
                weight_copy = child.weight.data.abs().clone()
                mask = weight_copy.gt(cfg_dict_refine["thre"]).float().to(args.device)
                cfg_dict_refine["prunedCount"] += mask.shape[0] - torch.sum(mask)
                # mask=0 的gamma和beta被置0
                child.weight.data.mul_(mask)
                child.bias.data.mul_(mask)
                cfg_dict_refine["cfg"].append(int(torch.sum(mask)))
                cfg_dict_refine["cfg_mask"].append(mask.clone())
                cfg_dict_refine["currentBlock"]["mainLastBN_mask"] = mask.clone()
            continue
        # 处理 BatchNorm2d
        elif isinstance(child, torch.nn.BatchNorm2d):
            weight_copy = child.weight.data.abs().clone()
            mask = weight_copy.gt(cfg_dict_refine["thre"]).float().to(args.device)
            cfg_dict_refine["prunedCount"] += mask.shape[0] - torch.sum(mask)
            # mask=0 的gamma和beta被置0
            child.weight.data.mul_(mask)
            child.bias.data.mul_(mask)
            cfg_dict_refine["cfg"].append(int(torch.sum(mask)))
            cfg_dict_refine["cfg_mask"].append(mask.clone())

            if name == "bn2":           # 主分支最后 BN
                cfg_dict_refine["currentBlock"]["mainLastBN_mask"] = mask.clone()
            elif name == "1":           # identity 分支最后 BN (downsample里的BN)
                cfg_dict_refine["currentBlock"]["idenLastBN_mask"] = mask.clone()

            # 对 bn 计数，与 cfg 索引一致
            cfg_dict_refine["bnCount"] += 1
        # 处理 select 层
        elif name == "select":
            cfg_dict_refine["currentBlock"]["select_mask"] = child.indexes
            if cfg_dict_refine["stageIdx"] == 0:
                if cfg_dict_refine["blockIdx"] == 0:
                    mask_res = cfg_dict_refine["currentBlock"]["mainLastBN_mask"]
                    cfg_dict_refine["select_mask_list"].append(mask_res)
                else:
                    mask_res = bitOr(
                        cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                        cfg_dict_refine["previousBlock"]["select_mask"])
                    cfg_dict_refine["select_mask_list"].append(mask_res)
            elif cfg_dict_refine["stageIdx"] in [1, 2, 3]:
                if cfg_dict_refine["blockIdx"] == 0:
                    mask_res = bitOr(
                        cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                        cfg_dict_refine["previousBlock"]["select_mask"])
                    cfg_dict_refine["select_mask_list"].append(mask_res)
                else:
                    mask_res = bitOr(
                        cfg_dict_refine["previousBlock"]["mainLastBN_mask"],
                        cfg_dict_refine["previousBlock"]["idenLastBN_mask"])
                    cfg_dict_refine["select_mask_list"].append(mask_res)
            cfg_dict_refine["select_cfg"].append(mask_res.sum())
        # 如果遇到一个 block
        elif child.__class__.__name__ == "BasicBlockWithSelect":
            traverse_refine(child, args, cfg_dict_refine, depth + 1, skipStem=False)

            # block结束后更新
            cfg_dict_refine["previousBlock"] = cfg_dict_refine["currentBlock"].copy()
            cfg_dict_refine["currentBlock"] = {"mainLastBN_mask": None, "idenLastBN_mask": None, "select_mask": None}

            cfg_dict_refine["blockIdx"] += 1
            continue
        # 如果遇到一个 stage (layer1, layer2, layer3, layer4)
        elif name in ["layer1", "layer2", "layer3", "layer4"]:
            traverse_refine(child, args, cfg_dict_refine, depth + 1, skipStem=False)

            # stage结束后更新
            cfg_dict_refine["stageIdx"] += 1
            cfg_dict_refine["blockIdx"] = 0  # 重置block计数
            continue

        # 递归进入其他子模块
        traverse_refine(child, args, cfg_dict_refine, depth + 1, skipStem=False)