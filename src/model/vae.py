import einops
import torch 
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.softplus = nn.Softplus()
        self.fc_mu = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, latent_dim) 
        )
        self.fc_logvar = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, latent_dim)
        )
        
    def reparameterize(self, dist: torch.distributions.MultivariateNormal):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def forward(
            self,
            input: torch.Tensor, # [batch_size, input_dim]
    ):  
        dtype = input.dtype
        mu = self.fc_mu(input) # [batch_size, latent_dim]
        logvar = self.fc_logvar(input) # [batch_size, latent_dim]
        mu = mu.to(torch.float32)
        logvar = logvar.to(torch.float32)
        scale = self.softplus(logvar) + 1e-6 # to ensure numberic stability
        scale_tril = torch.diag_embed(scale)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        z = self.reparameterize(dist)
        if self.training:
            # KLD loss
            std_normal = torch.distributions.MultivariateNormal(
                torch.zeros_like(z, device=z.device, dtype=torch.float32),
                scale_tril=torch.eye(z.shape[-1], device=z.device, dtype=torch.float32).unsqueeze(0).expand(z.shape[0], -1, -1),
            )
            kld_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        else:
            kld_loss = None
        
        return {
            "z": z.contiguous().to(dtype),
            "mu": mu.contiguous().to(dtype),
            "logvar": logvar.contiguous().to(dtype),
            "kld_loss": kld_loss
        }
    
    def sample(self, num_samples, device="cpu"):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        return z
    

class VQVAE(nn.Module):
    def __init__(
            self,
            input_dim: int,
            **vqvae_kwargs,
    ):
        super(VQVAE, self).__init__()
        self.input_dim = input_dim
        self.vq_layer = VectorQuantize(
            dim=input_dim,
            **vqvae_kwargs,
        )

    def forward(
            self,
            input: torch.Tensor, # [batch_size, sequence_length, input_dim]
    ):  
        input_dtype = input.dtype
        print(f"input dtype: {input_dtype}")
        quantized, _, vq_loss = self.vq_layer(input) # [batch_size, sequence_length, input_dim]
        quantized = quantized.to(input_dtype)

        return {
            "quantized": quantized,
            "vq_loss": vq_loss,
        }