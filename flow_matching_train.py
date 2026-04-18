import math
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
    steps: int = 5000
    batch_size: int = 1024
    lr: float = 2e-4
    hidden: int = 256
    n_layers: int = 4
    save_dir: str = "outputs/flow_matching/trained"
    save_dir_untrained: str = "outputs/flow_matching/untrained"
    sample_count: int = 2000
    ode_steps: int = 200
    ode_solver: str = "rk4"  # "euler" or "rk4"
    sample_before_train: bool = True
    save_grid_png: bool = True
    save_every: int = 20
    grid_cols: int = 8


class VelocityNet(nn.Module):
    def __init__(self, dim, hidden, n_layers):
        super().__init__()
        layers = []
        in_dim = dim + 1
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=1))


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


def sample_base(n, device):
    return torch.randn(n, 2, device=device)


def cfm_velocity(x0, x1, t):
    t = t.view(-1, 1)
    x_t = (1.0 - t) * x0 + t * x1
    v = x1 - x0
    return x_t, v


def train(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    model = VelocityNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for step in range(1, cfg.steps + 1):
        x1 = sample_mog(cfg.batch_size, cfg.device)
        x0 = sample_base(cfg.batch_size, cfg.device)
        t = torch.rand(cfg.batch_size, device=cfg.device)

        x_t, v_target = cfm_velocity(x0, x1, t)
        v_pred = model(x_t, t)
        loss = F.mse_loss(v_pred, v_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.6f}")

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "velocity_net.pt"))
    return model


def sample(model, cfg, tag=""):
    model.eval()
    with torch.no_grad():
        # Integrate the learned velocity field with the selected solver.
        x = sample_base(cfg.sample_count, cfg.device)
        t_grid = torch.linspace(0.0, 1.0, cfg.ode_steps + 1, device=cfg.device)
        dt = 1.0 / cfg.ode_steps
        frames = []
        t_vals = []

        for i in range(cfg.ode_steps):
            t = t_grid[i].expand(x.size(0))
            if cfg.ode_solver == "euler":
                v = model(x, t)
                x = x + dt * v
            elif cfg.ode_solver == "rk4":
                k1 = model(x, t)
                k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = model(x + dt * k3, t + dt)
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                raise ValueError("ode_solver must be 'euler' or 'rk4'")

            if cfg.save_grid_png and (i % cfg.save_every == 0 or i == cfg.ode_steps - 1):
                frames.append(x.detach().cpu().numpy())
                t_vals.append(t_grid[i].item())

        if cfg.save_grid_png and frames:
            save_grid(frames, t_vals, cfg, tag)

        return x.cpu().numpy()


def save_grid(frames, t_vals, cfg, tag=""):
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
    fig.suptitle("flow matching")

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.axis("off")
        if idx < n_frames:
            data = frames[idx]
            ax.scatter(data[:, 0], data[:, 1], s=4, alpha=0.6)
            ax.set_aspect("equal")
            t_val = t_vals[idx]
            ax.set_title(f"t = {t_val:.2f}", fontsize=9, pad=2)

    suffix = f"_{tag}" if tag else ""
    png_path = os.path.join(cfg.save_dir, f"flow_matching_grid{suffix}.png")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_samples(samples, cfg, tag=""):
    suffix = f"_{tag}" if tag else ""
    npy_path = os.path.join(cfg.save_dir, f"samples{suffix}.npy")
    np.save(npy_path, samples)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.6)
        plt.axis("equal")
        plt.axis("off")
        png_path = os.path.join(cfg.save_dir, f"samples{suffix}.png")
        plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    except Exception as exc:
        print(f"plotting failed {exc}")


if __name__ == "__main__":
    cfg = Config()
    if cfg.sample_before_train:
        untrained = VelocityNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
        original_dir = cfg.save_dir
        cfg.save_dir = cfg.save_dir_untrained
        samples = sample(untrained, cfg, tag="untrained")
        save_samples(samples, cfg, tag="untrained")
        cfg.save_dir = original_dir

    model = train(cfg)
    samples = sample(model, cfg, tag="trained")
    # save_samples(samples, cfg, tag="trained")
    print("done")
