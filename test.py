import os
import torch
import pandas as pd
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lib.model.resNet.resNet import ResNet
from tqdm import tqdm
from configs.test import parser


def get_fashion_mnist_labels(labels):
        """return text-labels of Fashion-mnist dataset"""
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]


def main():
    args = parser.parse_args()
    print(args)
    
    # pre-process
    # PIL -> torch.float32.Tensor
    # [0-255] -> [0, 1]
    # HWC -> CHW
    # resize to (256, 256)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))])
    # mnist_train: (tuple0, tuple1, ...)
    # tuplei: (image, label)
    # image: shape=(C, H, W)
    mnist_test = datasets.FashionMNIST(
        root=args.dataPath, train=False, transform=transform, download=False)
    testSize = len(mnist_test)
    print("test size: {}".format(testSize))
    print("sample 0 image shape: {}".format(mnist_test[0][0].shape))
    print("sample 0 label format: {}".format(mnist_test[0][1]))
    DataLoaderTest = DataLoader(
        mnist_test, batch_size=args.batchSize, num_workers=args.num_workers)

    resNet = ResNet()
    resNet.adaptMnist()
    # load checkpoint
    checkpointName = 'resNet_{}_{}_{}.pth'.format(
        args.checkSession, args.checkEpoch, args.checkPoint)
    print(">>> load checkpoint : {}".format(checkpointName))
    checkpointPath = os.path.join(
        args.checkpoint_dir, str(args.checkSession), str(checkpointName))
    resNet.loadCheckpoint(checkpointPath, args.device)
    # # check checkpoint path exists
    # if not os.path.exists(checkpointPath):
    #     raise FileNotFoundError(
    #         ">>> No checkpoint found at: {}".format(checkpointPath))
    # checkpoint = torch.load(checkpointPath, map_location=args.device)
    # resNet.load_state_dict(checkpoint['model_state_dict'])
    # print(">>> checkpoint loaded")

    criterion = torch.nn.CrossEntropyLoss()
    resNet.eval()
    resNet.to(args.device)

    iters_per_epoch = int(((testSize + args.batchSize - 1) / args.batchSize))
    res = torch.zeros(testSize, dtype=torch.int8)
    loss_avg_temp = 0.0
    acc_avg_temp = 0.0
    for step, (images, label) in tqdm(enumerate(DataLoaderTest)):
        # move data to device
        images = images.to(args.device)
        label = label.to(args.device)
        # train
        scores = resNet(images)
        loss = criterion(scores, label)
        label_hat = scores.argmax(dim=1)
        # store results
        res[step*args.batchSize : (step+1)*args.batchSize] = label_hat.cpu()

        with torch.no_grad():
            loss_avg_temp += loss.item()
            acc_avg_temp += (label_hat == label).float().mean().item()
    # compute loss and accuracy average
    loss_avg_temp /= iters_per_epoch
    acc_avg_temp /= iters_per_epoch
    # save results to file
    resultPath = os.path.join(
        args.res_dir, str(args.checkSession), 'predictions_{}_{}_{}.csv'.format(
        args.checkSession, args.checkEpoch, args.checkPoint))
    if not os.path.exists(os.path.dirname(resultPath)):
        os.makedirs(os.path.dirname(resultPath))

    df = pd.DataFrame({
        "sampleIdx": torch.arange(0, testSize),
        "predictionResult": res,
        "average loss": loss_avg_temp,
        "average accuracy": acc_avg_temp
    })
    df.to_csv(resultPath)

if __name__ == "__main__":
    main()