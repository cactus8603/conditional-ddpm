from glob import glob
import os 
import torch
b_t = torch.linspace(1e-4, 0.02, 1001)
a_t = 1 - b_t
ab_t = torch.cumprod(a_t, dim=0)

print(ab_t)