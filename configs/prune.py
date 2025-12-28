import argparse

parser = argparse.ArgumentParser(description='resNet slimming')
# test data
parser.add_argument('--dataPath', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--num_workers', type=int, default=12,
                    help='number of workers for data loading')
# checkpoint
parser.add_argument('--checkpoint_dir', type=str, default='./output/checkpoints/',
                    help='path to save checkpoints')
parser.add_argument('--checkSession', type=int, default=2,
                    help='session id for this training')
parser.add_argument('--checkEpoch', type=int, default=10,
                    help='epoch id for this training')
parser.add_argument('--checkPoint', type=int, default=469,
                    help='point id for this training')
# device
parser.add_argument('--device', type=str, default='cpu',
                    help='cpu or cuda')
# Prune settings
parser.add_argument('--percent', type=float, default=0.4,
                    help='scale sparse rate (default: 0.5)')