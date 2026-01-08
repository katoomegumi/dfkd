# diffusion_utils.py
import torch

# --- 全局常量 ---
TIMESTEPS = 100

# --- 函数 ---
def linear_beta_schedule(timesteps):
    """
    生成线性的 beta 调度表。
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def move_diffusion_schedules_to_device(device):
    """
    将所有全局的扩散调度张量移动到指定的设备。
    这个函数应该在主脚本开始时被调用一次。
    """
    global betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, recip_sqrt_alphas
    
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
    recip_sqrt_alphas = recip_sqrt_alphas.to(device)

def q_sample(x_start, t, noise=None):
    """
    前向加噪过程 (forward process)。
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # 确保张量在正确的设备上
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t.to('cpu')].to(x_start.device).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t.to('cpu')].to(x_start.device).view(-1, 1, 1, 1)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- 预计算的常量 ---
# 将这些常量放在模块级别，以便在导入时一次性计算好
betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
recip_sqrt_alphas = torch.sqrt(1.0 / alphas)