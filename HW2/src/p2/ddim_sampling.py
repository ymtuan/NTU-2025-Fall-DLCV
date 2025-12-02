import torch
import numpy as np

def get_time_steps(num_timesteps=50, total_steps=1000):
    """
    Generate uniform timesteps sequence.
    """
    # Calculate step size to get uniform steps
    step_size = total_steps // num_timesteps
    # Create sequence starting from 0 and add 1 to shift the first value from 0 to 1
    timesteps = np.asarray(list(range(0, total_steps, step_size))) + 1
    # Compute previous timesteps, starting with 0
    prev_timesteps = np.append(np.array([0]), timesteps[:-1])
    return timesteps, prev_timesteps

class DDIMSampler:
    def __init__(self, model, beta_start=0.0001, beta_end=0.02, num_timesteps=50):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = next(model.parameters()).device
        
        # Define beta schedule
        self.betas = torch.linspace(start=beta_start, end=beta_end, steps=1000, dtype=torch.float64, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        # Prepend 1.0 for the alpha_cumprod at t=0
        self.alpha_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_cumprod[:-1]], dim=0)

        # Precompute square roots for efficiency
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        
        # Get uniform timesteps
        self.timesteps, self.prev_timesteps = get_time_steps(num_timesteps=num_timesteps)
    
    def ddim_step(self, x_t, t, clip_denoised=True, eta=0.0):
        """
        Perform a single DDIM step, incorporating stochasticity via the eta parameter.
        """
        # Find the correct index of t in the timestep sequence
        t_idx = np.where(self.timesteps == t)[0][0]
        prev_t = self.prev_timesteps[t_idx]
        
        # Get precomputed coefficients for the current timestep t
        alpha_t = self.alpha_cumprod[t - 1]
        sqrt_alpha_t = self.sqrt_alpha_cumprod[t - 1]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumprod[t - 1]
        
        # Get precomputed coefficients for the previous timestep prev_t
        alpha_prev = self.alpha_cumprod[prev_t - 1] if prev_t > 0 else torch.tensor(1.0, device=self.device)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        
        # Predict noise using the model
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        epsilon_theta = self.model(x_t, t_tensor)
        
        # Predict x0 (the "predicted original image")
        x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
        # Calculate sigma_t for stochasticity, controlled by eta
        # If eta=0, sigma_t=0, resulting in a deterministic step (original DDIM)
        # If eta>0, introduces noise to the process
        sigma_t = 0.0
        if eta > 0:
            # The variance of the posterior q(x_{t-1}|x_t, x_0)
            variance = (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            sigma_t = eta * torch.sqrt(variance)

        # Get the direction pointing to x_t
        pred_dir_xt = torch.sqrt(1 - alpha_prev - sigma_t**2) * epsilon_theta
        
        # Generate random noise
        noise = torch.randn_like(x_t) if eta > 0 else 0.0

        # DDIM update step
        x_prev = sqrt_alpha_prev * x0_pred + pred_dir_xt + sigma_t * noise
        
        return x_prev
    
    @torch.no_grad()
    def ddim_sample(self, x, eta=0.0):
        """
        Generate samples using DDIM sampling.
        
        Args:
            x (torch.Tensor): The initial pure noise tensor of shape (batch_size, channels, height, width).
            eta (float): Controls the stochasticity of the sampling process. 
                         eta=0.0 is deterministic DDIM.
                         eta=1.0 is DDPM.
        """
        self.model.eval()
        x_t = x.to(self.device)
        
        # Sample loop, iterating backwards from the last timestep to the first
        for t in reversed(self.timesteps):
            x_t = self.ddim_step(x_t, t, clip_denoised=False, eta=eta)
        
        return x_t