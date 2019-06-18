import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Kaggle google landmark competition')

    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='learning rate')

    parser.add_argument('--optimizer',
                        type=str,
                        help='optimizer type',
                        default='SGD')

    parser.add_argument('--batch-size',
                        type=int,
                        help='mini-batch size',
                        default=512)

    parser.add_argument('--name',
                        required=True,
                        type=str,
                        help='session name')

    parser.add_argument('--crit', required=False, default='ce', const='focal', nargs='?', choices=['focal', 'ce'])

    parser.add_argument('--epochs',
                        required=True,
                        type=int,
                        help='number of epochs')

    parser.add_argument('--clip', required=False, default=None, type=float)
    parser.add_argument('--num-classes', required=True, default=None, type=int)

    parser.add_argument('--start-epoch', required=False, default=0, type=int, help='number of epoch to start with')

    parser.add_argument('--mixed',
                        action='store_true',
                        help='use mixed precision training')

    parser.add_argument('--resume', type=str, help='checkpoint to resume')

    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=4096)

    parser.add_argument('--num-workers', type=int, default=20)
    return parser


def get_make_index_parser():
    parser = argparse.ArgumentParser(description='Make index parser')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=20)
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--resume',type=str, required=True)
    parser.add_argument('--images-path', type=str, required=True)
    return parser


def get_retrieve_parser():
    parser = argparse.ArgumentParser(description='Retrieve parser')
    parser.add_argument('--index-path', type=str, required=True)
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--images-path', type=str, required=True)
    return parser
