"""
---
title: Denoising Diffusion Probabilistic Models (DDPM) training
summary: >
  Training code for
  Denoising Diffusion Probabilistic Model.
---

# [Denoising Diffusion Probabilistic Models (DDPM)](index.html) training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/experiment.ipynb)

This trains a DDPM based model on CelebA HQ dataset. You can find the download instruction in this
[discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
Save the images inside [`data/celebA` folder](#dataset_path).

The paper had used an exponential moving average of the model with a decay of $0.9999$. We have skipped this for
simplicity.

(obtained from: From: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/experiment.py)
"""
import os
import shutil

from typing import List

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torch.nn.utils import clip_grad_value_

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from noise import DenoiseDiffusion
from unet import UNet



def main():
    # Settings for restoring/creating experiment
    LOAD_CHECKPOINT = False # True, False
    UUID = 'AbeSaveTesting2'
    EXP = 'recurrent' # 'recurrent', 'residual'

    print(f'Status: Device is using GPU: {torch.cuda.is_available()}')

    # Create experiment
    experiment.create(
        name='diffuse',
        writers={'screen', 'labml'},
        uuid=UUID,
    )

    # Create configurations
    configs = Configs()
    # Set the model
    configs.convolutional_block = EXP
    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, {
        'dataset': 'MNIST',  # 'CIFAR10', 'CelebA' 'MNIST'
        'image_channels': 1,  # 3, 3, 1
        'epochs': 15,  # 100, 100, 5
    })
    # Initialize
    configs.init()

    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    # Start a new experiment
    if not LOAD_CHECKPOINT:
        # Reset the experiment
        if os.path.exists(f'{os.getcwd()}\logs\diffuse\{UUID}'):
            shutil.rmtree(f'{os.getcwd()}\logs\diffuse\{UUID}')
        # Start the experiment
        with experiment.start():
            configs.run()

    # Load previous experiment
    elif LOAD_CHECKPOINT:
        experiment.load(run_uuid=UUID)
        # Start the experiment (note: not using with experiment.start() when loading)
        with experiment.start():
            configs.run()


class Configs(BaseConfigs):
    """
    Class for holding configuration parameters for training a DDPM model.

    Attributes:
        device (torch.device):           Device on which to run the model.
        show (bool):                     Extract model information.
        eps_model (UNet):                U-Net model for the function `epsilon_theta`.
        diffusion (DenoiseDiffusion):    DDPM algorithm.
        image_channels (int):            Number of channels in the image (e.g. 3 for RGB).
        image_size (int):                Size of the image.
        n_channels (int):                Number of channels in the initial feature map.
        epochs (int):                    Number of training epochs.
        batch_size (int):                Batch size.
        clip (float):                    Magnitude of maximal gradients allowed.
        dropout (float):                 The probability for the dropout of units.
        channel_multipliers (List[int]): Number of channels at each resolution.
        is_attention (List[bool]):       Indicates whether to use attention at each resolution.
        convolutional_block (str):       Type of the convolutional block used
        n_steps (int):                   Number of time steps.
        n_samples (int):                 Number of samples to generate.
        learning_rate (float):           Learning rate.

        dataset (torch.utils.data.Dataset):         Dataset to be used for training.
        data_loader (torch.utils.data.DataLoader):  DataLoader for loading the data for training.
        optimizer (torch.optim.Adam):               Optimizer for the model.
    """

    # Device to train the model on.
    # [`DeviceConfigs`](https://docs.labml.ai/api/helpers.html#labml_helpers.device.DeviceConfigs)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device = DeviceConfigs()
    # Retrieve model information
    show: bool = True

    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 32
    # Number of channels in the initial feature map
    n_channels: int = 64  # 64 (Default: Ho et al.; Limit is VRAM)

    # Batch size
    batch_size: int = 64  # 64 (Default: Ho et al.; Limit is VRAM)
    # Number of training epochs
    epochs: int = 1000

    # Learning rate
    learning_rate: float = 2e-5
    # Set maximal gradient value
    clip: float = 1.0
    # Set the dropout probability
    dropout: float = 0.1
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]
    # Convolutional block type used in the UNet blocks. Possible options are 'residual' and 'recurrent'.
    convolutional_block: str = 'recurrent'

    # Number of time steps $T$ (with $T$ = 1_000 from Ho et al).
    n_steps: int = 1000  # 1000 (Default: Ho et al.)
    # Number of samples to generate
    n_samples: int = 16

    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam

    def init(self):
        """
        Initialize the model, dataset, and optimizer objects.
        """

        # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            dropout=self.dropout,
            is_attn=self.is_attention,
            conv_block=self.convolutional_block
        ).to(self.device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        # Show the number of params used by the model
        if self.show:
            pytorch_total_params = sum(p.numel() for p in self.eps_model.parameters())
            print(f'The total number of parameters are: {pytorch_total_params}')
        # Data augmentation
        self.dataset.data = self.augment(self.dataset.data)
        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

        # Image logging
        tracker.set_image("sample", True)

    def sample(self) -> None:
        """
        Generate samples from a trained Denoising Diffusion Probabilistic Model (DDPM).
        """

        with torch.no_grad():
            # Sample from the noise distribution at the final time step: x_T ~ p(x_T) = N(x_T; 0, I)
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            # Remove noise at each time step in reverse order (so remove noise for T steps)
            for t_ in monit.iterate('Sample', self.n_steps):
                # Get current time step
                t = self.n_steps - t_ - 1
                # Sample from the noise distribution at the current time step: x_{t-1} ~ p_theta(x_{t-1}|x_t)
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log the final denoised samples
            tracker.save('sample', x)

    def augment(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) == 3:
            img = img[:, None, :, :]

        transformations = [T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                           T.RandomAffine(degrees=90, translate=(0, 0.5), scale=(0.5, 0.5), shear=(0, 0.5))]

        for transform in transformations:
            img_trans = transform(img)
            img = torch.cat((img_trans, img), dim=0)
        return torch.squeeze(img)

    def train(self) -> None:
        """
        Train a Denoising Diffusion Probabilistic Model (DDPM) with the set dataloader.
        """

        # Iterate through the dataset
        for data in monit.iterate('Train', self.data_loader):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            data = data.to(self.device)

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            # Compute gradients
            loss.backward()
            # Clip model gradients
            clip_grad_value_(parameters=self.eps_model.parameters(), clip_value=self.clip)
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            tracker.save('loss', loss)

    def run(self):
        """
        ### Training loop
        """
        for _ in monit.loop(self.epochs):
            # Train the model
            self.train()
            # Sample some images
            self.sample()
            # New line in the console
            tracker.new_line()
            # Save the model
            experiment.save_checkpoint()


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size):
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


@option(Configs.dataset, 'MNIST')
def mnist_dataset(c: Configs):
    """
    Create MNIST dataset
    """
    return MNISTDataset(c.image_size)


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    """
    ### CIFAR10 dataset
    """

    def __init__(self, image_size):
        transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]

    @option(Configs.dataset, 'CIFAR10')
    def mnist_dataset(c: Configs):
        """
        Create CIFAR10 dataset
        """
        return CIFAR10Dataset(c.image_size)


if __name__ == '__main__':
    main()
