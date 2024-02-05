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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(54)  # best seed ever
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="ddpm_efdl", name=cfg['wandb_params']['run_name'])

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
    if cfg['augs']['use_flip']:
        all_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms = transforms.Compose(
        all_transforms
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=cfg['train_params']['batch_size'],
                            num_workers=cfg['train_params']['num_workers'],
                            shuffle=True)

    optim = hydra.utils.instantiate(cfg['optimizer'], params=ddpm.parameters())

    os.makedirs('samples', exist_ok=True)

    for i in range(num_epochs):
        train_epoch(ddpm, dataloader, optim, device, is_logging=True)
        generate_samples(ddpm, device, f"samples/{i:02d}.png", is_logging=True)
        if (i + 1) % 10 == 0:
            torch.save(ddpm.state_dict(), "ddpm.pt")


if __name__ == "__main__":
    main()
