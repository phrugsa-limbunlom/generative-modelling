import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    steps: int = 8000
    batch_size: int = 512
    lr: float = 2e-4
    hidden: int = 256
    n_layers: int = 4
    save_dir: str = "outputs/ebm/trained"
    save_dir_untrained: str = "outputs/ebm/untrained"
    sample_count: int = 2000
    cd_k: int = 30
    langevin_step_size: float = 0.05
    langevin_noise_scale: float = 0.1
    save_grid_png: bool = True
    save_every: int = 10
    grid_cols: int = 8
    sample_before_train: bool = True


class EnergyNet(nn.Module):
    def __init__(self, dim, hidden, n_layers):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_mog(n, device):
    centers = torch.tensor(
        [
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
            [2.0, 2.0],
            [-2.0, -2.0],
            [2.0, -2.0],
            [-2.0, 2.0],
        ],
        device=device,
    )
    idx = torch.randint(0, centers.size(0), (n,), device=device)
    x = centers[idx] + 0.25 * torch.randn(n, 2, device=device)
    return x


def langevin_step(x, model, step_size, noise_scale):
    x = x.detach().requires_grad_(True)
    energy = model(x).sum()
    grad = torch.autograd.grad(energy, x)[0]
    x = x - 0.5 * step_size * grad
    x = x + noise_scale * torch.randn_like(x)
    return x.detach()


def sample_negative(model, cfg, init=None):
    if init is None:
        x = torch.randn(cfg.batch_size, 2, device=cfg.device)
    else:
        x = init
    for _ in range(cfg.cd_k):
        x = langevin_step(x, model, cfg.langevin_step_size, cfg.langevin_noise_scale)
    return x


def train(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    model = EnergyNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for step in range(1, cfg.steps + 1):
        x_pos = sample_mog(cfg.batch_size, cfg.device)
        x_neg = sample_negative(model, cfg)

        energy_pos = model(x_pos).mean()
        energy_neg = model(x_neg).mean()
        loss = energy_pos - energy_neg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.6f}")

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "energy_net.pt"))
    return model


def sample(model, cfg, tag=""):
    model.eval()
    with torch.no_grad():
        x = torch.randn(cfg.sample_count, 2, device=cfg.device)
    frames = []
    steps = cfg.cd_k * 4
    for i in range(steps):
        x = langevin_step(x, model, cfg.langevin_step_size, cfg.langevin_noise_scale)
        if cfg.save_grid_png and (i % cfg.save_every == 0 or i == steps - 1):
            frames.append(x.detach().cpu().numpy())

    if cfg.save_grid_png and frames:
        save_grid(frames, cfg, tag)
    return x.cpu().numpy()


def save_grid(frames, cfg, tag=""):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"grid plotting disabled {exc}")
        return

    os.makedirs(cfg.save_dir, exist_ok=True)
    n_frames = len(frames)
    cols = max(1, cfg.grid_cols)
    rows = (n_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    fig.suptitle("ebm mcmc")

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    total = cfg.cd_k * 4
    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.axis("off")
        if idx < n_frames:
            data = frames[idx]
            ax.scatter(data[:, 0], data[:, 1], s=4, alpha=0.6)
            ax.set_aspect("equal")
            step = min((idx * cfg.save_every) + 1, total)
            ax.set_title(f"step = {step}", fontsize=9, pad=2)

    suffix = f"_{tag}" if tag else ""
    png_path = os.path.join(cfg.save_dir, f"ebm_grid{suffix}.png")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_samples(samples, cfg, tag=""):
    os.makedirs(cfg.save_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    npy_path = os.path.join(cfg.save_dir, f"ebm_samples{suffix}.npy")
    np.save(npy_path, samples)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.6)
        plt.axis("equal")
        plt.axis("off")
        png_path = os.path.join(cfg.save_dir, f"ebm_samples{suffix}.png")
        plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    except Exception as exc:
        print(f"plotting failed {exc}")


if __name__ == "__main__":
    cfg = Config()
    if cfg.sample_before_train:
        untrained = EnergyNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
        original_dir = cfg.save_dir
        cfg.save_dir = cfg.save_dir_untrained
        samples = sample(untrained, cfg, tag="untrained")
        save_samples(samples, cfg, tag="untrained")
        cfg.save_dir = original_dir

    model = train(cfg)
    samples = sample(model, cfg, tag="trained")
    # save_samples(samples, cfg, tag="trained")
    print("done")
