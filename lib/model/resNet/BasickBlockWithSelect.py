class BasicBlockWithSelect(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, cfg_select, stride=1, downsample=None):
        super(BasicBlockWithSelect, self).__init__()
        self.select = channel_selection(inplanes)
        # cfg [64, 64, 64] for resnet18
        #     [[previous block bn], [current block bn]]
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(cfg_select, cfg[1], stride)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[1], planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward_resBeforeSelect(self, x):
        residual = x
        out = self.select(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def forward_selectBeforeRes(self, x):
        out = self.select(x)
        residual = out.clone()
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def forward(self, x):
        # block with downsample
        if self.downsample is not None:
            return self.forward_selectBeforeRes(x)
        # block without downsample
        else:
            return self.forward_resBeforeSelect(x)