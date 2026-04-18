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
    steps: int = 6000
    batch_size: int = 1024
    lr: float = 2e-4
    hidden: int = 256
    n_layers: int = 4
    save_dir: str = "outputs/diffusion_score/trained"
    save_dir_untrained: str = "outputs/diffusion_score/untrained"
    sample_count: int = 2000
    t_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sample_before_train: bool = True
    save_grid_png: bool = True
    grid_cols: int = 8


class ScoreNet(nn.Module):
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


def make_schedule(cfg, device):
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.t_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample(x0, t_idx, alpha_bars):
    noise = torch.randn_like(x0)
    alpha_bar = alpha_bars[t_idx].view(-1, 1)
    x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
    return x_t


def score_target(x_t, x0, t_idx, alpha_bars):
    alpha_bar = alpha_bars[t_idx].view(-1, 1)
    return -(x_t - torch.sqrt(alpha_bar) * x0) / (1.0 - alpha_bar)


def train(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    model = ScoreNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    betas, alphas, alpha_bars = make_schedule(cfg, cfg.device)

    for step in range(1, cfg.steps + 1):
        x0 = sample_mog(cfg.batch_size, cfg.device)
        t_idx = torch.randint(0, cfg.t_steps, (cfg.batch_size,), device=cfg.device)
        t = (t_idx.float() + 1.0) / cfg.t_steps

        x_t = q_sample(x0, t_idx, alpha_bars)
        s_target = score_target(x_t, x0, t_idx, alpha_bars)
        s_pred = model(x_t, t)
        loss = F.mse_loss(s_pred, s_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.6f}")

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "score_net.pt"))
    return model


def p_sample_step(x_t, t_idx, betas, alphas, alpha_bars, model):
    beta_t = betas[t_idx]
    alpha_t = alphas[t_idx]
    alpha_bar_t = alpha_bars[t_idx]
    alpha_bar_prev = alpha_bars[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=x_t.device)
    t = torch.full((x_t.size(0),), (t_idx + 1) / len(betas), device=x_t.device)

    score = model(x_t, t)
    coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t + coef * score)

    if t_idx > 0:
        var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    return mean


def sample(model, cfg, tag=""):
    model.eval()
    betas, alphas, alpha_bars = make_schedule(cfg, cfg.device)

    t_grid = [i / 10.0 for i in range(11)]
    target_map = {int(round(t * cfg.t_steps)): t for t in t_grid}
    frames_by_step = {}

    with torch.no_grad():
        x = torch.randn(cfg.sample_count, 2, device=cfg.device)

        if cfg.save_grid_png and 0 in target_map:
            frames_by_step[0] = x.detach().cpu().numpy()

        step_count = 0
        for t_idx in reversed(range(cfg.t_steps)):
            x = p_sample_step(x, t_idx, betas, alphas, alpha_bars, model)
            step_count += 1
            if cfg.save_grid_png and step_count in target_map:
                frames_by_step[step_count] = x.detach().cpu().numpy()

        if cfg.save_grid_png and frames_by_step:
            ordered_idx = sorted(frames_by_step.keys())
            frames = [frames_by_step[i] for i in ordered_idx]
            t_vals = [target_map[i] for i in ordered_idx]
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
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0), constrained_layout=True)
    # Set the title to indicate trained or untrained
    if tag == "untrained":
        title = "Score-Based Diffusion Model - Sample Trajectories (untrained)"
    else:
        title = "Score-Based Diffusion Model - Sample Trajectories (trained)"
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.08)

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
        if idx < n_frames:
            data = frames[idx]
            # Use a light academic blue (e.g., #4F8CC9)
            ax.scatter(data[:, 0], data[:, 1], s=24, alpha=0.7, color='#4F8CC9', edgecolor='k', linewidth=0.5)
            ax.set_aspect("equal")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_xticks(np.arange(-4, 5, 2))
            ax.set_yticks(np.arange(-4, 5, 2))
            ax.grid(True, linestyle='--', color='gray', alpha=0.7, linewidth=1.0)
            t_val = t_vals[idx]
            ax.set_title(f"Step {idx} (t={t_val:.3f})", fontsize=12, pad=6)
            ax.tick_params(axis='both', which='major', labelsize=9, length=4)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_title("")
            ax.grid(False)

    plt.subplots_adjust(top=0.90, wspace=0.25, hspace=0.25)
    suffix = f"_{tag}" if tag else ""
    png_path = os.path.join(cfg.save_dir, f"diffusion_score_grid{suffix}.png")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    cfg = Config()
    if cfg.sample_before_train:
        untrained = ScoreNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
        original_dir = cfg.save_dir
        cfg.save_dir = cfg.save_dir_untrained
        sample(untrained, cfg, tag="untrained")
        cfg.save_dir = original_dir

    model = train(cfg)
    sample(model, cfg, tag="trained")
    print("done")
