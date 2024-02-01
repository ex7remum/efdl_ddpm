import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os

import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(54)  # best seed ever
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="ddpm_efdl", name=cfg['run_name'])
    wandb.log_artifact("config/config.yaml")

    num_epochs = cfg['train_params']['epochs']

    ddpm = DiffusionModel(
        eps_model=UnetModel(cfg['model']['img_channels'],
                            cfg['model']['img_channels'],
                            hidden_size=cfg['model']['hidden_size']),
        betas=cfg['model']['betas'],
        num_timesteps=cfg['model']['num_timesteps'],
    )
    ddpm.to(device)
    wandb.watch(ddpm)

    all_transforms = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if cfg['augs']['rand_flip']:
        all_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms = transforms.Compose(
        all_transforms
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=cfg['train_params']['batch_size'],
                            num_workers=cfg['train_params']['num_workers'],
                            shuffle=True)

    if cfg['train_params']['optimizer'] == 'adam':
        optim = torch.optim.Adam(ddpm.parameters(), lr=cfg['train_params']['learning_rate'])
    elif cfg['train_params']['optimizer'] == 'sgd':
        optim = torch.optim.SGD(ddpm.parameters(), lr=cfg['train_params']['learning_rate'],
                                momentum=cfg['train_params']['momentum'])
    else:
        raise NotImplementedError

    os.makedirs('samples', exist_ok=True)

    for i in range(num_epochs):
        train_epoch(ddpm, dataloader, optim, device, is_logging=True)
        generate_samples(ddpm, device, f"samples/{i:02d}.png", is_logging=True)


if __name__ == "__main__":
    main()
