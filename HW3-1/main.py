import os
import argparse
import tensorflow as tf

from src.data_loader import GenerateDataSet
from model import BaseLineModel

def parse_args():
    project_dir = os.path.dirname(__file__)
    project_data_dir = os.path.join(project_dir, 'data')
    anime_datasets_dir_path = os.path.join(project_data_dir, 'AnimeDataset/faces/')

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default=anime_datasets_dir_path, help='')
    parser.add_argument('--img_width', type=int, default=64, help='')
    parser.add_argument('--img_height', type=int, default=64, help='')
    parser.add_argument('--img_channel', type=int, default=3, help='')

    parser.add_argument('--batch_size', type=int, default=5, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='')
    parser.add_argument('--beta1', type=float, default=0.5, help='')
    parser.add_argument('--max_epoch', type=int, default=10, help='')
    parser.add_argument('--noise_dim', type=int, default=100, help='')

    parser.add_argument('--mode', type=str, default='train', help='train or infer')

    return parser.parse_args()

def main(args):

    model = BaseLineModel(
        args.img_width, args.img_height, args.img_channel,
        args.batch_size, args.learning_rate, args.beta1, args.max_epoch, args.noise_dim
    )

    if args.mode == 'train':
        model.train(args.image_dir)
    elif args.mode == 'infer':
        pass
        # model.infer(args.image_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)