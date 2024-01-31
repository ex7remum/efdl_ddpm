import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os

from hparams import config
import wandb

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


def main(device: str):
    torch.manual_seed(54)  # best seed ever
    wandb.init(config=config, project="ddpm_efdl", name="init run")
    num_epochs = config['epochs']

    ddpm = DiffusionModel(
        eps_model=UnetModel(config['img_channels'],
                            config['img_channels'],
                            hidden_size=config['hidden_size']),
        betas=config['betas'],
        num_timesteps=config['num_timesteps'],
    )
    ddpm.to(device)
    wandb.watch(ddpm)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                     num_workers=config['num_workers'],
                                     shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=config['learning_rate'])

    os.makedirs('samples', exist_ok=True)

    for i in range(num_epochs):
        train_epoch(ddpm, dataloader, optim, device, is_logging=True, num_epoch=i)
        generate_samples(ddpm, device, f"samples/{i:02d}.png", is_logging=True)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)
