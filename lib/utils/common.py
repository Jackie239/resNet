import torch
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


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


def test(model, args, loadPercent=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))])
    mnist_test = datasets.FashionMNIST(
        root=args.dataPath, train=False, transform=transform, download=False)

    testSize = int(len(mnist_test)*loadPercent)
    indices = list(range(testSize))
    mnist_test = Subset(mnist_test, indices)

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