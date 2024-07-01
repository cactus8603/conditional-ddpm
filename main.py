import torch
import numpy as np
import random
import argparse
import os
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from utils import train_one_epoch
from model import ContextUNet
from diffusion import ConditionalDiffusionModel
from dataset import TrainImageDataset, ValImageDataset

def create_parser():
    parser = argparse.ArgumentParser()

    ### data path
    # parser.add_argument("--config_path", default="config.yaml", nargs='?', help="path to config file")
    parser.add_argument("--data_path", default='./dataset', type=str, help='') 
    # parser.add_argument("--sample_set", default='./sample/test_style', type=str, help='')
<<<<<<< HEAD
    parser.add_argument("--save_path", default='./result/n_down_4', type=str, help='path to save model and tbwriter')
    parser.add_argument("--load_model_path", default='/code/conditional-ddpm/result/test/model_82_0.012_.pth', type=str, help='')
=======
    parser.add_argument("--save_path", default='./result/test', type=str, help='path to save model and tbwriter')
    # parser.add_argument("--load_model_path", default='/code/conditional-ddpm/result/add_c_loss_noise/model_98_0.006_.pth', type=str, help='')
>>>>>>> 4b972075974ad16762fae68982ee60f1d1ab156b

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

    # model = ContextUNet()

    ### DataSet ### fromat: data_path // train/val // input/target
    train_dataset = TrainImageDataset(os.path.join(arg.data_path, 'train/target'), os.path.join(arg.data_path, 'train/input'))
<<<<<<< HEAD
    # val_dataset = ValImageDataset(os.path.join(arg.data_path, 'val/target'), os.path.join(arg.data_path, 'val/input'))
=======
    # val_dataset = ValImageDataset(os.path.join(arg.data_path, 'val/input'), os.path.join(arg.data_path, 'val/target'))
>>>>>>> 4b972075974ad16762fae68982ee60f1d1ab156b

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

    ### load model
    denoiser = ContextUNet().to(device)
<<<<<<< HEAD
    denoiser = torch.load(arg.load_model_path)
=======
    # denoiser = torch.load(arg.load_model_path)
>>>>>>> 4b972075974ad16762fae68982ee60f1d1ab156b
    diffusion = ConditionalDiffusionModel(model=denoiser, device=device).to(device)

    # setting optim
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=arg.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=50)
    scaler = amp.GradScaler()

    
<<<<<<< HEAD
    pbar = trange(arg.epochs)
    tmp_loss = 99
    for epoch in pbar:
=======
    pbar = tqdm(arg.epochs)
    tmp_loss = 999
    for epoch in range(arg.epochs):
>>>>>>> 4b972075974ad16762fae68982ee60f1d1ab156b
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
        # for i, (target_img, condition_img) in enumerate(valloader):
        #     target_img, condition_img = target_img.to(device), condition_img.to(device)
        #     diffusion.save_generated_samples_into_folder(n_samples=arg.batch_size, condition=condition_img, folder_path=arg.save_path, epoch=i)

        #     if i==1: break

        pbar.update(1)
        # break
        pbar.desc = "epoch:{}, loss:{:.4f}".format(epoch, loss)
        

        tb_writer.add_scalar('loss', loss, epoch)
        if loss < tmp_loss:
            tmp_loss = loss
            if epoch > 10:
                save_path = os.path.join(arg.save_path, "model_{}_{:.3f}_.pth".format(epoch, loss))
                torch.save(denoiser, save_path)
            
