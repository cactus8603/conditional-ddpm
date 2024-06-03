import torch
import numpy as np
import random
import argparse
import os
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import train_one_epoch
from model import ContextUNet
from diffusion import diffusion
from dataset import TrainImageDataset, ValImageDataset

def create_parser():
    parser = argparse.ArgumentParser()

    ### data path
    # parser.add_argument("--config_path", default="config.yaml", nargs='?', help="path to config file")
    parser.add_argument("--data_path", default='./dataset', type=str, help='') 
    # parser.add_argument("--sample_set", default='./sample/test_style', type=str, help='')
    parser.add_argument("--save_path", default='./result', type=str, help='path to save model and tbwriter')
    # parser.add_argument("--json_file", default='./cfgs/font_classes_50.json', type=str, help='')

    ### training setting
    parser.add_argument("--lr", default=1e-2, type=float, help='learning rate')
    parser.add_argument("--epochs", default=200, type=int, help='total epoch')
    parser.add_argument("--batch_size", default=32, type=int, help='total classes')
    
    parser.add_argument("--num_workers", default=6, type=int, help='')
    parser.add_argument("--timesteps", default=1000, type=int, help='')
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
    arg = create_parser()

    # init random seed
    init(arg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(device)

    if not os.path.exists(arg.save_path):
        os.mkdir(arg.save_path)
    tb_writer = SummaryWriter(arg.save_path)

    model = ContextUNet()

    ### DataSet ### fromat: data_path // train/val // input/target
    train_dataset = TrainImageDataset(os.path.join(arg.data_path, 'train/input'), os.path.join(arg.data_path, 'train/target'))
    val_dataset = ValImageDataset(os.path.join(arg.data_path, 'val/input'), os.path.join(arg.data_path, 'val/target'))

    ### DataLoader ###
    trainloader = DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        pin_memory=True,
        num_workers=arg.num_workers,
    )
    valloader = DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        pin_memory=True,
        num_workers=arg.num_workers,
    )

    # setting optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=50)
    scaler = amp.GradScaler()

    ### load model
    denoiser = ContextUNet()
    diffusion = diffusion(model=denoiser)

    pbar = tqdm(trainloader)
    tmp_loss = 999
    for epoch in range(arg.epochs):
        # denoiser, optimizer, dataloader, scaler, scheduler, device, arg
        # training 
        loss = train_one_epoch(
            diffusion = diffusion, 
            optimizer = optimizer, 
            dataloader = trainloader,                        
            scaler = scaler,
            scheduler = scheduler,
            epoch = epoch,
            device = device,
            arg = arg
        )

        ### ToDo: evaluation

        pbar.desc = "epoch:{}, loss:{:.4f}".format(epoch, loss)
        pbar.update(1)

        tb_writer.add_scalar('loss', loss, epoch)
        if loss < tmp_loss:
            save_path = os.path.join(arg.save_path, "model_{}_{:.3f}_.pth".format(epoch, loss))
            torch.save(model, save_path)
            tmp_loss = loss

