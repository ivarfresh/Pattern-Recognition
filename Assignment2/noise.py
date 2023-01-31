"""
---
title: Denoising Diffusion Probabilistic Models (DDPM)
summary: >
  PyTorch implementation and tutorial of the paper
  Denoising Diffusion Probabilistic Models (DDPM).
---

(obtained from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/__init__.py)
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn.functional as f
import torch.utils.data
from torch import nn
from utils import gather


class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        Initialize a DenoiseDiffusion object.

        Args:
        - eps_model: This is \epsilon_\theta(x_t, t). PyTorch module (with parameters theta) representing the denoising
                     process, which is a function that maps the final latent state back to the observation space (e.g.,
                     the space of clean images).
        It takes the final latent state x_t and produces some output. In our case, this is the UNet.
        - n_steps: t: the number of steps in the diffusion process.
        - device: the device (e.g., CPU or GPU) on which the model's parameters and data should be stored.
        """
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.device = device

        # Create an increasing variance schedule for the diffusion process.
        self.beta = self.get_named_beta_schedule().to(self.device)
        # Compute the complementary values of beta.
        self.alpha = 1. - self.beta
        # Compute the cumulative product of alpha.
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Set the variance of the diffusion process to beta.
        self.sigma2 = self.beta


    def get_named_beta_schedule(self) -> torch.Tensor:
        """
        Get a pre-defined beta schedule for the given name.
        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """

        # Linear schedule from Ho et al., extended to work for any number of diffusion steps.
        scale = 1000 / self.n_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.n_steps, dtype=torch.float32)

    def betas_for_alpha_bar(self, alpha_bar: get_named_beta_schedule, max_beta: int = 0.999) -> torch.Tensor:
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        Args
        - num_diffusion_timesteps: the number of betas to produce.
        - alpha_bar: a lambda that takes an argument t from 0 to 1 and
                     produces the cumulative product of (1-beta) up to that
                     part of the diffusion process.
        -  max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        """

        betas = []
        for i in range(self.n_steps):
            t1 = i / self.n_steps
            t2 = (i + 1) / self.n_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the q(x_t|x_0) distribution. In other words, compute the distribution of the final latent state given the
        initial latent state and the current step in the diffusion process.

        The distribution is a Gaussian with mean equal to a weighted average of the initial latent state and the current
        latent state and variance equal to a fixed value.

        The q(x_t|x_0) distribution is defined as:
        q(x_t|x_0) = 𝒩(x_t; √(α̅ₜ ) * x_0, (1 - α̅ₜ ) * I)
        (in latex: q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big) )

        Args:
        - x0: a tensor representing the initial latent state.
        - t: a tensor representing the current step in the diffusion process.

        Returns:
        A tuple containing the mean and variance of the distribution.

        """
        # Gather the value of alpha_bar for the current step, and then compute sqrt(α̅ₜ ) * x_0
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # The variance is then: (1 - α̅ₜ ) * I
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        Sample from the distribution of the final latent state given the initial latent state and the current step
        in the diffusion process. The distribution is a Gaussian with mean equal to a weighted average of the initial
         latent state and the current latent state and variance equal to a fixed value.

        Args:
        - x0: a tensor representing the initial latent state.
        - t: a tensor representing the current step in the diffusion process.
        - eps: an optional tensor representing noise to be added to the sample.
               If not provided, noise will be generated by sampling from a standard normal distribution.

        Returns:
        A tensor representing a sample from the distribution.

        """
        # Get the distribution (so, $q(x_t|x_0)$) of the final latent state given the initial latent state and the
        # current step in the diffusion process.
        mean, var = self.q_xt_x0(x0, t)
        # If no noise is provided, generate noise by sampling from a standard normal distribution (so from N(I,0)).
        if eps is None:
            eps = torch.randn_like(x0)
        # Sample from that distribution $q(x_t|x_0)$ with Gaussian noise.
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from the distribution defined by the denoising process. So, sample from pₜ(𝑥ₜ₋₁|𝑥ₜ) as follows:
        pₜ(𝑥ₜ₋₁ | 𝑥ₜ) = 𝒩(𝑥ₜ₋₁; μₜ(𝑥ₜ, t), σ²ₜ𝐈)
        μₜ(𝑥ₜ, t) = 1/√αₜ (𝑥ₜ - εₜ(𝑥ₜ, t)βₜ/√(1-ᾱₜ))

        Args:
        - xt: a tensor representing the final latent state.
        - t: a tensor representing the current step in the diffusion process.

        Returns:
        A tensor representing a sample from the distribution.
        """

        # Get the output of the denoising process, which is a function that maps the
        # final latent state back to the observation space (e.g., the space of clean images).
        eps_theta = self.eps_model(xt, t)
        # Get the value of alpha_bar at the current step in the diffusion process.
        alpha_bar = gather(self.alpha_bar, t)
        # Get the value of αₜ at the current step in the diffusion process.
        alpha = gather(self.alpha, t)
        # Calculate the mean of the distribution.
        mean = 1 / (alpha ** 0.5) * (xt - (1 - alpha) / (1 - alpha_bar) ** 0.5 * eps_theta)
        # Get the variance of the distribution at the current step in the diffusion process.
        var = gather(self.sigma2, t)

        # Sample from the distribution.
        eps = torch.randn(xt.shape,
                          device=xt.device)
        return mean + (var ** 0.5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Calculate the loss for the given initial latent state. The loss is calculated as the expectation of the squared
        difference between noise ε and the output of a model εₜ given input √ᾱₜ𝑥₀ + √1 - ᾱₜε at time t.
        Simplified Loss: L_simple(θ) = 𝔼ₜ,𝑥₀,ε[|| ε - εₜ(𝑥₀√ᾱₜ + ε√(1-ᾱₜ), t) ||²]

        Args:
        - x0: a tensor representing the initial latent state.
        - noise: a tensor representing noise to be used in sampling. If not provided, noise will be generated randomly
                 from a normal distribution with mean 0 and variance 1.

        Returns:
        A tensor representing the loss.
        """

        # Get batch size
        batch_size = x0.shape[0]
        # Get random t for each sample in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device,
                          dtype=torch.long)  # Randint gives  unfiormly distributed ints

        # Generate noise if it was not provided
        if noise is None:
            # Generate noise from a normal distribution with mean 0 and variance 1
            noise = torch.randn_like(x0)

        # Sample xₜ for q(xₜ | x₀)
        xt = self.q_sample(x0, t, eps=noise)
        # Get εₜ(x₀√ᾱₜ  + ε√1-ᾱₜ, t)
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return f.mse_loss(noise, eps_theta)
