import argparse


parser = argparse.ArgumentParser(description='Train a resNet')
# data
parser.add_argument('--dataPath', type=str, default='./data/preTrainedModel/resnet18-5c106cde.pth',
                    help='path to dataset')
parser.add_argument('--shuffle', type=bool, default=True,
                    help='shuffle the dataset')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers for data loading')
# model
parser.add_argument('--pretrained', type=bool, default=True,
                    help='Use pretrained resNet model')
# training
parser.add_argument('--batchSize', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
# device
parser.add_argument('--device', type=str, default='cpu',
                    help='cpu or cuda')
# log
parser.add_argument('--use_tensorboard', type=bool, default=True,
                    help='whether to use tensorboard to log training status')
parser.add_argument('--tensorboard_dir', type=str, default='./output/tensorboard/',
                    help='path to save tensorboard logs')
parser.add_argument('--log_interval', type=int, default=10,
                    help='number of steps between logging training status')
# checkpoint
parser.add_argument('--checkpoint_dir', type=str, default='./output/checkpoints/',
                    help='path to save checkpoints')
parser.add_argument('--session', type=int, default=1,
                    help='session id for this training')
parser.add_argument('--save_interval', type=int, default=2,
                    help='number of epochs between saving model checkpoints')
