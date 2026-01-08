import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.weightsFormat = "IMAGENET1K_V1"
        self.model = resnet18(weights=None)
        self.preProcess = ResNet18_Weights.IMAGENET1K_V1.transforms()


    def forward(self, input):
        return self.model(input)


    def loadPretrainedModel(self, pretrainedModelPath):
        print(">>> Loading Pretrained ResNet18 model.")
        # load resNet model from torchvision.models
        # load from local path if pretrained is True
        if os.path.exists(pretrainedModelPath):
            self.model.load_state_dict(torch.load(pretrainedModelPath))
        # download 
        else:
            return resnet18(weights=self.weightsFormat)


    def adaptMnist(self):
        print(">>> Initializing ResNet model.")
        print(">>> conv1 weight shape: (3, 64) - > (1, 64)")
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        print(">>> fc weight shape: (512, 1000) - > (512, 10)")
        self.model.fc = nn.Linear(in_features=512, out_features=10, bias=True)


    # def loadCheckpoint(self, checkpointPath, device):
    #     # check checkpoint path exists
    #     if not os.path.exists(checkpointPath):
    #         raise FileNotFoundError(
    #             ">>> No checkpoint found at: {}".format(checkpointPath))
    #     checkpoint = torch.load(checkpointPath, map_location=device)
    #     cleaned = {k.replace("model.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    #     self.model.load_state_dict(cleaned)
    #     print(">>> checkpoint loaded")