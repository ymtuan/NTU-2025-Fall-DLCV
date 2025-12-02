import numpy as np

def get_beta_scheduler(beta_start=0.0001, beta_end=0.02, n_timestep=1000):
    """Linear beta schedule from beta_start to beta_end"""
    return np.linspace(start=beta_start, stop=beta_end, num=n_timestep, dtype=np.float64)
