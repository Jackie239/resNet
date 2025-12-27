import os
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lib.model.resNet.resNet import ResNet
from tqdm import tqdm
from configs.train import parser
import torch.nn as nn


def get_fashion_mnist_labels(labels):
    """return text-labels of Fashion-mnist dataset"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def updateBN(model, lambdaSparsity):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(lambdaSparsity*torch.sign(m.weight.data))  # L1


def main():
    args = parser.parse_args()
    print(args)
    if args.use_tensorboard:
        log_dir = os.path.join(args.tensorboard_dir, str(args.session))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

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
    mnist_train = datasets.FashionMNIST(
        root=args.dataPath, train=True, transform=transform, download=False
    )
    trainSize = len(mnist_train)
    print("training size: {}".format(trainSize))
    # print("test size: {}".format((len(mnist_test))))
    print("sample 0 image shape: {}".format(mnist_train[0][0].shape))
    print("sample 0 label format: {}".format(mnist_train[0][1]))
    DataLoaderTrain = DataLoader(mnist_train, batch_size=args.batchSize,
                                shuffle=args.shuffle, num_workers=args.num_workers)

    # create model
    resNet = ResNet()
    resNet.adaptMnist()
    resNet.train()
    resNet.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resNet.parameters(), lr=args.lr)

    iters_per_epoch = int(((trainSize + args.batchSize - 1) / args.batchSize))
    for epoch in tqdm(range(args.num_epochs), desc="Epochs", position=0):
    # for epoch in range(args.num_epochs):
        if args.use_tensorboard:
            loss_avg_temp = 0.0
            acc_avg_temp = 0.0
        for step, (images, label) in tqdm(enumerate(DataLoaderTrain)):
            # move data to device
            images = images.to(args.device)
            label = label.to(args.device)
            # train
            scores = resNet(images)
            loss = criterion(scores, label)
            optimizer.zero_grad()
            loss.backward()
            if args.useNS:
                updateBN(resNet, args.lambdaSparsity)
            optimizer.step()
            # log
            if args.use_tensorboard:
                # compute loss and accuracy
                with torch.no_grad():
                    loss_avg_temp += loss.item()
                    acc_avg_temp += (scores.argmax(dim=1) == label).float().mean().item()
                if (step+1) % args.log_interval == 0:
                    # loss average
                    loss_avg_temp /= args.log_interval
                    acc_avg_temp /= args.log_interval
                    writer.add_scalar('train/loss', loss_avg_temp, epoch*iters_per_epoch+step+1)
                    writer.add_scalar('train/accuracy', acc_avg_temp, epoch*iters_per_epoch+step+1)
                    loss_avg_temp = 0
                    acc_avg_temp = 0
        # save model checkpoint
        if (epoch+1) % args.save_interval == 0:
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            modelSavePath = os.path.join(
                args.checkpoint_dir, str(args.session),
                "resNet_{}_{}_{}.pth".format(args.session, epoch+1, step+1))
            checkpoint = {'epoch': epoch + 1,
                        'step': step + 1,
                        'model_state_dict': resNet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "loss": loss.item()}
            torch.save(checkpoint, modelSavePath)



if __name__ == "__main__":
    main()