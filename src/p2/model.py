import torch
import numpy as np
from utils import beta_scheduler
from UNet import UNet
from torchvision.utils import save_image
import os

class DDIM:
    def __init__(self, model, n_timesteps=1000, n_steps=250, eta=0.0, device='cuda'):
        """
        Initialize the DDIM sampler with full timestep range.
        """
        self.model = model
        self.n_timesteps = n_timesteps
        self.n_steps = n_steps
        self.eta = eta
        self.device = device

        # Load beta schedule
        beta = beta_scheduler(n_timestep=self.n_timesteps).to(self.device).float()
        self.beta = beta
        self.alpha = 1.0 - beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_cumprod[:-1]], dim=0)

        # Precompute square roots
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1.0 / self.alpha_cumprod)
        self.sqrt_alpha_cumprod_prev = torch.sqrt(self.alpha_cumprod_prev)
        self.sqrt_one_minus_alpha_cumprod_prev = torch.sqrt(1.0 - self.alpha_cumprod_prev)

        step_size = self.n_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.n_timesteps, step_size))) + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        self.timesteps = ddim_timestep_seq
        self.prev_timesteps = ddim_timestep_prev_seq
        

    def ddim_step(self, x_t, t_idx, clip_denoised=True):
        """
        Perform a single DDIM step.
        t_idx is the index into the precomputed arrays (0 to 999).
        """
        prev_t_idx = max(0, t_idx - 1)  # Ensure prev_t_idx is within bounds

        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t_idx]
        sqrt_alpha_cumprod_prev = self.sqrt_alpha_cumprod_prev[prev_t_idx]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t_idx]

        # Predict noise
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
        epsilon_theta = self.model(x_t, t_tensor)
        # print(f"t_idx: {t_idx}, prev_t_idx: {prev_t_idx}, x_t range: [{x_t.min()}, {x_t.max()}], epsilon range: [{epsilon_theta.min()}, {epsilon_theta.max()}]")

        # Predict x_0
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * epsilon_theta) * self.sqrt_recip_alpha_cumprod[t_idx]
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Direction pointing to x_t
        dir_xt = sqrt_one_minus_alpha_cumprod_t * epsilon_theta

        # Compute x_{t-1}
        coeff = (sqrt_alpha_cumprod_prev / sqrt_alpha_cumprod_t) * (x_t - dir_xt) + dir_xt
        x_prev = coeff

        if self.eta > 0:
            sigma_t = self.eta * torch.sqrt((1 - self.alpha_cumprod[t_idx]) / (1 - self.alpha_cumprod_prev[prev_t_idx]) * (1 - self.alpha_cumprod_prev[prev_t_idx] / self.alpha_cumprod[t_idx]))
            x_prev += sigma_t * torch.randn_like(x_prev)

        return x_prev

    def sample(self, batch_size, channels, height, width, ground_truth_noise=None, save_intermediate=False, save_dir='samples'):
        if save_intermediate:
            os.makedirs(save_dir, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            if ground_truth_noise is not None:
                x_t = ground_truth_noise.to(self.device)
            else:
                x_t = torch.randn(batch_size, channels, height, width, device=self.device)

            print(f"Initial x_t range: [{x_t.min()}, {x_t.max()}]")
            if save_intermediate:
                x_t_denorm = (x_t + 1) / 2
                x_t_denorm = torch.clamp(x_t_denorm, 0, 1)
                save_image(x_t_denorm, os.path.join(save_dir, "step_0.png"))

            # Reverse timesteps from n_steps-1 to 0
            for i, t_idx in enumerate(reversed(range(self.n_steps))):
                x_t = self.ddim_step(x_t, t_idx, clip_denoised=True)
                if i % (self.n_steps // 5) == 0:
                    print(f"Step {i}, x_t range: [{x_t.min()}, {x_t.max()}]")
                if save_intermediate and i % (self.n_steps // 5) == 0:
                    x_t_denorm = (x_t + 1) / 2
                    x_t_denorm = torch.clamp(x_t_denorm, 0, 1)
                    save_image(x_t_denorm, os.path.join(save_dir, f"step_{i}.png"))

            print(f"Final x_t range: [{x_t.min()}, {x_t.max()}]")
            return x_t