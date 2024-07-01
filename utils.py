import torch
import math
import os
from tqdm import tqdm
from torch.nn import functional as F
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast

from diffusion import ConditionalDiffusionModel

def prepare_data(input_images, target_images, noise_level):
    noise = torch.randn_like(input_images)
    noisy_input_images = input_images + noise_level * noise
    return noisy_input_images, noise, target_images

def get_ddpm_noise_schedule(timesteps, device, initial_beta=1e-4, final_beta=0.02):
        """Generate DDPM noise schedule.

        Args:
            timesteps (int): Number of timesteps.
            initial_beta (float): Initial value of beta.
            final_beta (float): Final value of beta.
            device (torch.device): Device to generate the schedule on.

        Returns:
            tuple: Tuple containing a_t, b_t, and ab_t.
                - a_t (torch.Tensor): Alpha values.
                - b_t (torch.Tensor): Beta values.
                - ab_t (torch.Tensor): Cumulative alpha values.
        """
        b_t = torch.linspace(initial_beta, final_beta, timesteps + 1, device=device)
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        return a_t, b_t, ab_t

def perturb_input(x, t, noise, ab_t):
        """Perturbs given input
        i.e., Algorithm 1, step 5, argument of epsilon_theta in the article
        """
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise

def train_one_epoch(diffusion, optimizer, dataloader, scaler, scheduler, epoch, device, arg):
    # model.train()
    # diffusion = diffusion(model=denoiser)
    # loss_function = torch.nn.MSELoss()
    
    a_t, b_t, ab_t = get_ddpm_noise_schedule(arg.timesteps, device=device, initial_beta=1e-4, final_beta=0.02)

    # pbar = tqdm(dataloader)
    optimizer.zero_grad()
    running_loss = 0.0
    
    for i, (target_img, condition_img) in enumerate(dataloader):
        target_img, condition_img = target_img.to(device), condition_img.to(device)

        with amp.autocast():
            # pred noise
            x_pert, predict_noise, x_denoise, loss = diffusion(target_img, condition_img)
            # get loss
            # loss = F.mse_loss(predict_noise, noise)
        
        scaler.scale(loss).backward()

        if (i + 1) % arg.accumulation_step == 0:
            # upgrade params
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * arg.accumulation_step
        # if i==10 : break
        
    
    diffusion.save_tensor_images(target_img, x_pert, x_denoise, epoch, arg.save_path)

    diffusion.save_generated_samples_into_folder(n_samples=arg.batch_size, condition=condition_img, folder_path=arg.save_path, epoch=epoch)

    # diffusion.simple_sample(n_samples=arg.batch_size, epoch=epoch, save_dir=arg.save_path, condition=condition_img, )
            
    scheduler.step()
    average_loss = running_loss / len(dataloader)
    # pbar.desc = "epoch:{}, loss:{:.5f}".format(epoch, average_loss)
    # pbar.update(1)

    return average_loss


