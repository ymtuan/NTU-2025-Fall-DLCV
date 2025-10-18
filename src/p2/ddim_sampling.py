import torch
import numpy as np

def get_time_steps(num_timesteps=50, total_steps=1000):
    """
    Generate the specific uniform timesteps as required: [981, 961, 941, ..., 1].
    """
    return np.linspace(total_steps - 19, 1, num_timesteps, dtype=np.int64)

class DDIMSampler:
    def __init__(self, model, beta_start=0.0001, beta_end=0.02, num_timesteps=50):
        self.model = model
        self.num_timesteps = num_timesteps
        
        # Define the full 1000-step beta/alpha schedule
        self.betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Get the specific 50 timesteps for the sampling sequence
        self.timesteps = get_time_steps(num_timesteps=self.num_timesteps, total_steps=1000)
    
    @torch.no_grad()
    def ddim_sample(self, x, eta=0.0):
        # Ensure model is in evaluation mode
        self.model.eval()
        device = next(self.model.parameters()).device
        
        x_t = x.to(device)
        
        # Move schedule tensors to the correct device once
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Iterate through the custom timestep sequence (e.g., from 981 down to 1)
        for i in range(len(self.timesteps)):
            # Current timestep 't' from our sequence (e.g., 981)
            t = self.timesteps[i]
            
            # Previous timestep 't-1' from our sequence (e.g., 961)
            # For the last step, the previous timestep is 0
            prev_t = self.timesteps[i + 1] if i < len(self.timesteps) - 1 else 0
            
            # Get the pre-calculated sqrt(alpha_cumprod) values for t and prev_t
            # Note: We use t-1 for 0-based indexing.
            sqrt_alpha_t = sqrt_alphas_cumprod[t - 1]
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t - 1]
            
            # Handle the final step where prev_t = 0
            if prev_t == 0:
                sqrt_alpha_prev = torch.tensor(1.0, device=device)
                sqrt_one_minus_alpha_prev = torch.tensor(0.0, device=device)
            else:
                sqrt_alpha_prev = sqrt_alphas_cumprod[prev_t - 1]
                sqrt_one_minus_alpha_prev = sqrt_one_minus_alphas_cumprod[prev_t - 1]

            # 1. Predict the noise `ε_θ(x_t, t)` using the U-Net model
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            noise_pred = self.model(x_t, t_tensor)

            # 2. **BUG FIX**: Use the direct mathematical formula for the update step.
            # This avoids the errors from incorrectly combining intermediate terms.
            # Formula: x_{t-1} = (ᾱ_{t-1}/ᾱ_t)^0.5 * x_t + ((1-ᾱ_{t-1})^0.5 - (ᾱ_{t-1}(1-ᾱ_t)/ᾱ_t)^0.5) * ε_θ
            
            term1_coeff = sqrt_alpha_prev / sqrt_alpha_t
            term2_coeff = sqrt_one_minus_alpha_prev - (sqrt_alpha_prev * sqrt_one_minus_alpha_t) / sqrt_alpha_t
            
            # Calculate x_{t-1}
            x_prev = term1_coeff * x_t + term2_coeff * noise_pred
            
            # Update x_t for the next iteration in the loop
            x_t = x_prev
            
        return x_t