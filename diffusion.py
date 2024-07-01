import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from model import ContextUNet

# ConditionalDiffusionModel
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, model, device):
        super(ConditionalDiffusionModel, self).__init__()
        self.timesteps = 1000
        self.model = model
        self.device = device
        self.inference_transform = lambda x: (x + 1)/2

    def forward(self, x, condition_image):
        self.model.train()
        ### 實作將圖片加噪，並contextUNet去噪

        B, _, _, _ = x.shape

        a_t, b_t, ab_t = self.get_ddpm_noise_schedule(self.timesteps)

        ### 
        noise = torch.rand_like(x)
        t = torch.randint(1, self.timesteps+1, (x.shape[0],), device=x.device)
        # t_tmp = int(t)
        # print(t)

        # add noise
        x_pert = self.perturb_input(x, t, noise, ab_t)

        # pred noise
        predict_noise = self.model(x_pert, t / self.timesteps, condition_image=condition_image)
        x_denoised = self.get_x_unpert(x_pert, t, predict_noise, ab_t)
        loss1 = F.mse_loss(predict_noise, noise)

        # pred_img_test
        # for denoise_step in range(self.timesteps-1, 0, -1):
            # x_denoised = (self.get_x_unpert(x_pert, denoise_step, predict_noise, ab_t))

        # pred_img_test_2
        # x_denoised = self.get_x0(x_pert, t, predict_noise, target_img,  ab_t)
            
        # x_denoised_trans = self.inference_transform(x_denoised)
        # loss2 = F.mse_loss(x_denoised_trans, target_img)

        loss = loss1
        
        # return x_pert, predict_noise, x_denoised, loss
        return x_pert, predict_noise, self.get_x_unpert(x_pert, t, predict_noise, ab_t), loss


    def get_ddpm_noise_schedule(self, timesteps, initial_beta=1e-4, final_beta=0.02):
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
        b_t = torch.linspace(initial_beta, final_beta, timesteps + 1, device=self.device)
        a_t = 1 - b_t
        ab_t = torch.cumprod(a_t, dim=0)
        return a_t, b_t, ab_t
    
    def perturb_input(self, x, t, noise, ab_t):
        """Perturbs given input
        i.e., Algorithm 1, step 5, argument of epsilon_theta in the article
        """
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise
    
    def get_x_unpert(self, x_pert, t, pred_noise, ab_t):
        """Removes predicted noise pred_noise from perturbed image x_pert"""
        return (x_pert - (1 - ab_t[t, None, None, None]).sqrt() * pred_noise) / ab_t.sqrt()[t, None, None, None]
    
    def save_tensor_images(self, x_orig, x_noised, x_denoised, epoch, save_dir):
        """Saves given tensors as a single image"""
        fpath = os.path.join(save_dir, 'orig_noise_denoise_{}.jpg'.format(epoch))
        inference_transform = lambda x: (x + 1)/2
        save_image([make_grid(inference_transform(img.detach())) for img in [x_orig, x_noised, x_denoised]], fpath)

    

    @torch.no_grad()
    def simple_sample(self, n_samples, epoch, save_dir, condition=None, inference_transform=lambda x: (x+1)/2, ):
        self.model.eval()

        a_t, b_t, ab_t = self.get_ddpm_noise_schedule(self.timesteps)

        self.model.eval()
        samples = torch.randn(n_samples, self.model.in_channels, self.model.height, self.model.width, device=self.device)

        for t in range(self.timesteps, 0, -1):
            z = torch.randn_like(samples) if t > 1 else 0
            pred_noise = self.model(samples, torch.tensor([t/self.timesteps], device=self.device), condition)
            samples = self.denoise_add_noise(samples, t, pred_noise, a_t, b_t, ab_t, z)
        
        samples = inference_transform(samples.detach().cpu())
        samples = torch.clamp(samples, 0, 1)

        for i in range(n_samples):
            fpath = os.path.join(save_dir, 'sample_{}_{}.jpg'.format(epoch, i))
            save_image(samples[i], fpath)
        # fpath = os.path.join(save_dir, 'sample_{}.jpg'.format(epoch))
        # save_image((inference_transform(samples.detach().cpu())), fpath)

        # return inference_transform(samples.detach().cpu())

    
    @torch.no_grad()
    def sample_ddpm(self, n_samples, condition=None, timesteps=None, 
                    beta1=None, beta2=None, save_rate=20, inference_transform=lambda x: (x+1)/2):
        """Returns the final denoised sample x0,
        intermediate samples xT, xT-1, ..., x1, and
        times tT, tT-1, ..., t1
        """

        a_t, b_t, ab_t = self.get_ddpm_noise_schedule(self.timesteps)

        
        self.model.eval()
        samples = torch.randn(n_samples, self.model.in_channels, 
                              self.model.height, self.model.width, 
                              device=self.device)
        intermediate_samples = [samples.detach().cpu()] # samples at T = timesteps
        t_steps = [self.timesteps] # keep record of time to use in animation generation
        for t in range(self.timesteps, 0, -1):
            # print(f"Sampling timestep {t}", end="\r")
            # if t % 50 == 0: print(f"Sampling timestep {t}")

            z = torch.randn_like(samples) if t > 1 else 0
            pred_noise = self.model(samples, torch.tensor([t/self.timesteps], device=self.device)[:, None, None, None], condition)
            samples = self.denoise_add_noise(samples, t, pred_noise, a_t, b_t, ab_t, z)
            
            if t % save_rate == 1 or t < 8:
                intermediate_samples.append(inference_transform(samples.detach().cpu()))
                t_steps.append(t-1)
        return intermediate_samples[-1], intermediate_samples, t_steps

    def denoise_add_noise(self, x, t, pred_noise, a_t, b_t, ab_t, z):
        """Removes predicted noise from x and adds gaussian noise z
        i.e., Algorithm 2, step 4 at the ddpm article
        """
        noise = b_t.sqrt()[t]*z
        denoised_x = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
        return denoised_x + noise
    
    def save_generated_samples_into_folder(self, n_samples, condition, folder_path, epoch):
        """Save DDPM generated inputs into a specified directory"""
        samples, _, _ = self.sample_ddpm(n_samples, condition)
        for i, sample in enumerate(samples):
            if not os.path.exists(os.path.join(folder_path, str(epoch))):
                os.makedirs(os.path.join(folder_path, str(epoch)))
            save_image(sample, os.path.join(folder_path, str(epoch), f"image_{epoch}_{i}.jpeg"))

 

    # # add noise
    # def q_sample(self, x0, t, alphas_cumprod):
    #     noise = torch.randn_like(x0)
    #     sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
    #     sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])
    #     xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    #     return xt, noise

    # # denoise
    # def p_sample(self, xt, t, target_image, betas):
    #     pred_noise = self(xt, t, target_image)
    #     alpha_t = 1 - betas[t]
    #     alpha_t_cumprod = torch.prod(alpha_t)
    #     mean = (xt - betas[t] * pred_noise) / torch.sqrt(alpha_t)
    #     noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
    #     xt_prev = mean + torch.sqrt(betas[t]) * noise
    #     return xt_prev


    # def sample(model, target_image, alphas_cumprod, betas, timesteps):
    #     model.eval()
    #     with torch.no_grad():
    #         x = torch.randn_like(target_image)  # 开始于纯噪声图像
    #         for t in reversed(range(timesteps)):
    #             x = model.p_sample(x, t, target_image, betas, alphas_cumprod)
    #     return x
