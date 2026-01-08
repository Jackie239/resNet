import torch
import os

def loadCheckpoint(model, checkpointPath, device, strict=True, replace=True):
    # check checkpoint path exists
    if not os.path.exists(checkpointPath):
        raise FileNotFoundError(
            ">>> No checkpoint found at: {}".format(checkpointPath))
    checkpoint = torch.load(checkpointPath, map_location=device)
    if replace:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    else:
        state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=strict)
    print(">>> checkpoint loaded")