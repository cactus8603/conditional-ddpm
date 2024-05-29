import torch
import numpy as np
import random
import argparse

def create_parser():
    parser = argparse.ArgumentParser()

    ### data path
    # parser.add_argument("--config_path", default="config.yaml", nargs='?', help="path to config file")
    parser.add_argument("--data_path", default='/code/Font/fonts_50/val_byFont', type=str, help='') # /code/Font/fonts_50/val_byFont # /code/Font/fonts_50/byFont
    # parser.add_argument("--style_dir", default='/code/Font/fonts_50/byFont', type=str, help='')
    parser.add_argument("--sample_set", default='./sample/test_style', type=str, help='')
    parser.add_argument("--json_file", default='./cfgs/font_classes_50.json', type=str, help='')

    ### training setting
    parser.add_argument("--lr", default=5e-6, type=float, help='learning rate')
    parser.add_argument("--epoch", default=2000, type=int, help='total epoch')
    parser.add_argument("--n_classes", default=173, type=int, help='total classes')
    parser.add_argument("--n_steps", default=1000, type=int, help='')
    parser.add_argument("--n_samples", default=1, type=int, help='Number of samples to generate, only 1 now')
    parser.add_argument("--accumulation_step", default=4, type=int, help='')
    parser.add_argument("--seed", default=8603, type=int, help='init random seed')

    # parser.add_argument("", default=, type=, help='')

    args = parser.parse_args()
    return args

# set seed
def init(seed):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = create_parser()

    # init random seed
    init(args.seed)