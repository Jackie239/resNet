import argparse


parser = argparse.ArgumentParser(description='Train a resNet')
# data
parser.add_argument('--dataPath', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--num_workers', type=int, default=12,
                    help='number of workers for data loading')
# testing
parser.add_argument('--batchSize', type=int, default=64,
                    help='input batch size for training')
# device
parser.add_argument('--device', type=str, default='cuda',
                    help='cpu or cuda')
# log
parser.add_argument('--use_tensorboard', type=bool, default=True,
                    help='whether to use tensorboard to log training status')
parser.add_argument('--tensorboard_dir', type=str, default='./output/tensorboard/',
                    help='path to save tensorboard logs')
parser.add_argument('--log_interval', type=int, default=10,
                    help='number of steps between logging,' \
                    ' for test, it equals checkPoint')
# checkpoint
parser.add_argument('--checkpoint_dir', type=str, default='./output/checkpoints/',
                    help='path to save checkpoints')
parser.add_argument('--checkSession', type=int, default=1,
                    help='session id for this training')
parser.add_argument('--checkEpoch', type=int, default=2,
                    help='epoch id for this training')
parser.add_argument('--checkPoint', type=int, default=469,
                    help='point id for this training')
# results
parser.add_argument('--res_dir', type=str, default='./output/predictions/',
                    help='path to save output predictions')