import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import wandb

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str,
                is_logging: bool = False, num_epoch: int = 0):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for i, (x, _) in enumerate(pbar):
        if i == 0 and is_logging:
            grid = make_grid(x[:8], normalize=True, value_range=(-1, 1), nrow=4)
            images = wandb.Image(grid)
            wandb.log({"Input examples": images}, step=num_epoch+1)

        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        if is_logging:
            metrics = {'train_loss': train_loss}
            wandb.log(metrics, step=num_epoch * len(dataloader.dataset) + (i + 1) * wandb.config["batch_size"])
        pbar.set_description(f"loss: {loss_ema:.4f}")


def generate_samples(model: DiffusionModel, device: str, path: str, is_logging: bool = False, num_epoch: int = 0):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device=device)
        grid = make_grid(samples, normalize=True, value_range=(-1, 1), nrow=4)
        save_image(grid, path)
        if is_logging:
            wandb.log({"Samples": wandb.Image(path)}, step=num_epoch+1)
