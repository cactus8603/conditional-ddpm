import torch
import math
from tqdm import tqdm
from torch.nn import functional as F
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast

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

# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     Cosine beta schedule for diffusion process.
#     Args:
#         timesteps (int): Number of diffusion steps.
#         s (float): Small constant for numerical stability.
#     Returns:
#         betas (torch.Tensor): Beta values for each timestep.
#     """
#     steps = torch.arange(timesteps + 1, dtype=torch.float64)
#     alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return betas

# def diffusion_loss(model, input_images, target_images, t, noise_level):
#     noisy_input_images, noise, target_images = prepare_data(input_images, target_images, noise_level)
#     pred_noise = model(noisy_input_images, t, target_images)
#     loss = F.mse_loss(pred_noise, noise)
#     return loss

def train_one_epoch(model, optimizer, dataloader, epoch, scaler, scheduler, device, arg):
    model.train()
    loss_function = torch.nn.MSELoss()
    optimizer.zero_grad()

    a_t, b_t, ab_t = get_ddpm_noise_schedule(arg.timesteps, device=device, initial_beta=1e-4, final_beta=0.02)

    pbar = tqdm(dataloader)
    running_loss = 0.0
    
    for i, (input_img, target_img) in enumerate(dataloader):
        input_img, target_img = input_img.to(device), target_img.to(device)
        
        ### 
        noise = torch.rand_like(input_img)
        t = torch.randint(0, 1000, (input_img.shape[0],), device=input_img.device)
        x_pert = perturb_input(input_img, t, noise, ab_t)

        with amp.autocast():
            # pred noise
            predict_noise = model(x_pert, t / arg.timesteps, )
            # get loss
            loss = F.mse_loss(predict_noise, noise)
        
        scaler.scale(loss).backward()

        if (i + 1) % arg.accumulation_steps == 0:
            # upgrade params
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * arg.accumulation_steps
            
    scheduler.step()
    average_loss = running_loss / len(dataloader)
    pbar.desc = "epoch:{}, loss:{:.5f}".format(epoch, average_loss)
    pbar.update(1)

    return average_loss

    # 

    # print(f'Epoch [{epoch+1}], Loss: {avg_loss:.4f}')

    # noise_level = torch.tensor(0.1)  # Example noise level
    # noisy_input_img, noise, target_img = prepare_data(input_img, target_img, noise_level)
    # pred_noise = model(noisy_input_img, t, target_img)
    # loss = F.mse_loss(pred_noise, noise)
    # loss.backward()
    # optimizer.step()
    # return loss.item()

def train(model, dataloader, optimizer, arg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=50)
    scaler = amp.GradScaler()
    pbar = tqdm(dataloader)
    for epoch in range(arg.num_epochs):

        # training model, optimizer, dataloader, epoch, scaler, scheduler, device, arg
        loss = train_one_epoch(
            model = model, 
            optimizer = optimizer, 
            dataloader = dataloader, 
            epoch = epoch, 
            scaler = scaler,
            scheduler = scheduler,
            device = device,
            arg = arg
        )

        pbar.desc = "epoch:{}, loss:{:.4f}".format(epoch, loss)
        pbar.update(1)

