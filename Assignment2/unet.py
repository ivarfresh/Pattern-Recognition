"""
---
title: U-Net model for Denoising Diffusion Probabilistic Models (DDPM)
summary: >
  UNet model for Denoising Diffusion Probabilistic Models (DDPM)
---

# U-Net model for [Denoising Diffusion Probabilistic Models (DDPM)](index.html)

This is a [U-Net](../../unet/index.html) based model to predict noise
$\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$.

U-Net is a gets it's name from the U shape in the model diagram.
It processes a given image by progressively lowering (halving) the feature map resolution and then
increasing the resolution.
There are pass-through connection at each resolution.

![U-Net diagram from paper](../../unet/unet.png)

This implementation contains a bunch of modifications to original U-Net (residual blocks, multi-head attention)
 and also adds time-step embeddings $t$.
"""

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

from labml_helpers.module import Module

from collections import OrderedDict
from torch import nn

class Swish(Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    Embeddings for the time step in the input data.

    Parameters
    ----------
    n_channels : int
        The number of dimensions in the embedding.

    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        """
        Create sinusoidal position embeddings similar to those from the transformer.

        PE[1][t, i] = sin(t / 10^(i / (d - 1)))
        PE[2][t, i] = cos(t / 10^(i / (d - 1)))

        where d is half_dim.

        Parameters
        ----------
        t : torch.Tensor
            Tensor of shape (batch_size, 1) representing the time step.

        Returns
        -------
        emb : torch.Tensor
            Tensor of shape (batch_size, n_channels) representing the time embeddings.

        """
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb

class ResidualBlock(Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(x)
        h = self.act1(h)
        h = self.norm1(x)
        # Add time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(h)
        h = self.act2(h)
        h = self.norm2(h)

        # Add the shortcut connection and return
        return h + self.shortcut(x)

class RecurrentBlock(nn.Module):

    scale = 1  # scale of the bottleneck convolution channels
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, recurrent=1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `recurrent` is the number of times the input is recurrently passed. Block output is treated as new input.
        """
        super().__init__()

        self.recurrent = recurrent

        # Group normalization and the first convolution layer
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),  bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        # Group normalization and the first convolution layer
        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.act1 = Swish()

        # Group normalization and the first convolution layer
        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.act2 = Swish()

        # Group normalization and the first convolution layer
        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=(1, 1), bias=False)
        self.act3 = Swish()

        self.output = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels * self.scale)

        # need BatchNorm for each time step for training to work well
        for r in range(self.recurrent):
            setattr(self, f'norm1_{r}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{r}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{r}', nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h = self.conv_input(x)

        for r in range(self.recurrent):
            if r == 0:
                shortcut = self.norm_skip(self.skip(h))
            else:
                shortcut = h

            # First convolution layer in block 't'
            h = self.conv1(h)
            h = getattr(self, f'norm1_{r}')(h)
            h = self.act1(h)

            h += self.time_emb(t)[:, :, None, None]

            # Second convolution layer in block 't'
            h = self.conv2(h)
            h = getattr(self, f'norm2_{r}')(h)
            h = self.act2(h)

            # Third convolution layer in block 't'
            h = self.conv3(h)
            h = getattr(self, f'norm3_{r}')(h)

            # Skip connection
            h += shortcut
            h = self.act3(h)

        return self.output(h)

class AttentionBlock(Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 has_attn: bool, conv_block: str = 'residual'):
        super().__init__()
        if conv_block == 'residual':
            self.re = ResidualBlock(in_channels, out_channels, time_channels)
        elif conv_block == 'recurrent':
            self.re = RecurrentBlock(in_channels, out_channels, time_channels, 2)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.re(x, t)
        x = self.attn(x)
        return x


class UpBlock(Module):
    """
    ### Up block

    This combines `Residual/RecurrentBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 has_attn: bool, conv_block: str = 'residual'):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        if conv_block == 'residual':
            self.re = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        elif conv_block == 'recurrent':
            self.re = RecurrentBlock(in_channels + out_channels, out_channels, time_channels, 2)

        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.re(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(Module):
    """
    ### Middle block

    It combines a `Residual/RecurrentBlock`, `AttentionBlock`, followed by another `Residual/RecurrentBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, conv_block: str = 'residual'):
        super().__init__()
        if conv_block == 'residual':
            self.re1 = ResidualBlock(n_channels, n_channels, time_channels)
            self.attn = AttentionBlock(n_channels)
            self.re2 = ResidualBlock(n_channels, n_channels, time_channels)
        if conv_block == 'recurrent':
            self.re1 = RecurrentBlock(n_channels, n_channels, time_channels, 2)
            self.attn = AttentionBlock(n_channels)
            self.re2 = RecurrentBlock(n_channels, n_channels, time_channels, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.re1(x, t)
        x = self.attn(x)
        #x = self.re2(x, t)
        return x


class Upsample(Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    # Note to self (Abe): this ConvTranspose2d does all the upscaling. Describing how it does so,
    # is something for the methods

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2, conv_block: str = 'residual'):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()
        # Convolutional block type for UNet blocks.
        if not (conv_block in ['residual', 'recurrent']):
            raise NotImplementedError(f'unknown convolutional block type: {conv_block}.'
                                      f' Possible options are: "residual" and "recurrent".')

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)


        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))#, conv_block))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, conv_block)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))#, conv_block))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)
        # Recurrent: (64, 1024,1, 1)
        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                
                #print(f'S size no condat:              {s.size()}')
                #print(f'X size no concat:              {x.size()}')
                
                x = torch.cat((x, s), dim=1)
                
                #print(f'S size yes concat: {s.size()}')
                #print(f'X concat S:                    {x.size()}')
                #
                x = m(x, t)
                
                #print(f'UP time emb (t) + img proj (x) {x.size()}')
                
# =============================================================================
#attempt to fix it
#                 if x.size(dim =2) == s.size(dim=2):
#                     x = torch.cat((x, s), dim=1)
#                 elif x.size(dim =2) != s.size(dim=2):
#                     print(s[3])
#                     s = s.expand_as(x)
#                 
# =============================================================================
        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

