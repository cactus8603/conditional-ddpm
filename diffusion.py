import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ContextUNet

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, model):
        super(ConditionalDiffusionModel, self).__init__()
        self.timesteps = 1000
        self.model = model

    def forward(self, x, t, target_image):
        ### 實作將圖片加噪，並contextUNet去噪
        a_t, b_t, ab_t = self.get_ddpm_noise_schedule(self.timesteps)

    
        pass

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
    
    

    # add noise
    def q_sample(self, x0, t, alphas_cumprod):
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    # denoise
    def p_sample(self, xt, t, target_image, betas):
        pred_noise = self(xt, t, target_image)
        alpha_t = 1 - betas[t]
        alpha_t_cumprod = torch.prod(alpha_t)
        mean = (xt - betas[t] * pred_noise) / torch.sqrt(alpha_t)
        noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
        xt_prev = mean + torch.sqrt(betas[t]) * noise
        return xt_prev


    def sample(model, target_image, alphas_cumprod, betas, timesteps):
        model.eval()
        with torch.no_grad():
            x = torch.randn_like(target_image)  # 开始于纯噪声图像
            for t in reversed(range(timesteps)):
                x = model.p_sample(x, t, target_image, betas, alphas_cumprod)
        return x
