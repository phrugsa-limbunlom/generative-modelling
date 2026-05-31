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
    save_dir: str = "outputs/diffusion_cfg/trained"
    save_dir_untrained: str = "outputs/diffusion_cfg/untrained"
    sample_count: int = 2000
    t_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sample_before_train: bool = True
    save_grid_png: bool = True
    save_every: int = 20
    grid_cols: int = 8
    num_classes: int = 8
    cond_drop_prob: float = 0.1
    guidance_scale: float = 3.0
    sample_class: int = -1
    use_cfg: bool = True
    save_all_steps: bool = False
    save_step_png: bool = False
    grid_steps: int = 10


class ConditionalNoiseNet(nn.Module):
    def __init__(self, dim, hidden, n_layers, num_classes):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.label_emb = nn.Embedding(num_classes + 1, hidden)

        layers = []
        in_dim = dim + hidden
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, labels):
        t = t.view(-1, 1)
        t_emb = self.time_mlp(t)
        label_emb = self.label_emb(labels)
        cond = t_emb + label_emb
        return self.net(torch.cat([x, cond], dim=1))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_mog_with_labels(n, device):
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
    labels = torch.randint(0, centers.size(0), (n,), device=device)
    x = centers[labels] + 0.25 * torch.randn(n, 2, device=device)
    return x, labels


def make_schedule(cfg, device):
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.t_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample(x0, t_idx, alpha_bars):
    noise = torch.randn_like(x0)
    alpha_bar = alpha_bars[t_idx].view(-1, 1)
    x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
    return x_t, noise


def apply_cond_dropout(labels, num_classes, drop_prob):
    if drop_prob <= 0.0:
        return labels
    drop_mask = torch.rand(labels.shape, device=labels.device) < drop_prob
    null_labels = torch.full_like(labels, num_classes)
    return torch.where(drop_mask, null_labels, labels)


def train(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    model = ConditionalNoiseNet(
        dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers, num_classes=cfg.num_classes
    ).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    betas, alphas, alpha_bars = make_schedule(cfg, cfg.device)

    for step in range(1, cfg.steps + 1):
        x0, labels = sample_mog_with_labels(cfg.batch_size, cfg.device)
        t_idx = torch.randint(0, cfg.t_steps, (cfg.batch_size,), device=cfg.device)
        t = (t_idx.float() + 1.0) / cfg.t_steps

        x_t, noise = q_sample(x0, t_idx, alpha_bars)
        labels_drop = apply_cond_dropout(labels, cfg.num_classes, cfg.cond_drop_prob)
        noise_pred = model(x_t, t, labels_drop)
        loss = F.mse_loss(noise_pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            print(f"step {step} loss {loss.item():.6f}")

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "noise_net_cfg.pt"))
    return model


def p_sample_step(x_t, t_idx, betas, alphas, alpha_bars):
    beta_t = betas[t_idx]
    alpha_t = alphas[t_idx]
    alpha_bar_t = alpha_bars[t_idx]
    alpha_bar_prev = alpha_bars[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=x_t.device)
    return beta_t, alpha_t, alpha_bar_t, alpha_bar_prev


def predict_eps(model, x, t, labels, cfg):
    if not cfg.use_cfg:
        return model(x, t, labels)

    null_labels = torch.full_like(labels, cfg.num_classes)
    eps_uncond = model(x, t, null_labels)
    eps_cond = model(x, t, labels)
    return eps_uncond + cfg.guidance_scale * (eps_cond - eps_uncond)


def sample(model, cfg, tag=""):
    model.eval()
    betas, alphas, alpha_bars = make_schedule(cfg, cfg.device)

    t_grid = [i / cfg.grid_steps for i in range(cfg.grid_steps + 1)]
    target_map = {int(round(t * cfg.t_steps)): t for t in t_grid}
    frames_by_step = {}

    def step_to_t(step_idx):
        if step_idx == 0:
            return 0.0
        return (cfg.t_steps - step_idx + 1) / cfg.t_steps

    with torch.no_grad():
        x = torch.randn(cfg.sample_count, 2, device=cfg.device)

        if cfg.sample_class < 0:
            labels = torch.randint(0, cfg.num_classes, (x.size(0),), device=cfg.device)
        else:
            labels = torch.full((x.size(0),), cfg.sample_class, device=cfg.device, dtype=torch.long)

        if cfg.save_all_steps or (cfg.save_grid_png and 0 in target_map):
            frames_by_step[0] = x.detach().cpu().numpy()

        step_count = 0
        for t_idx in reversed(range(cfg.t_steps)):
            t = torch.full((x.size(0),), (t_idx + 1) / cfg.t_steps, device=cfg.device)
            beta_t, alpha_t, alpha_bar_t, alpha_bar_prev = p_sample_step(
                x, t_idx, betas, alphas, alpha_bars
            )

            eps = predict_eps(model, x, t, labels, cfg)
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps)

            if t_idx > 0:
                var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

            step_count += 1
            if cfg.save_all_steps or (cfg.save_grid_png and step_count in target_map):
                frames_by_step[step_count] = x.detach().cpu().numpy()

        if cfg.save_all_steps:
            save_step_samples(frames_by_step, cfg, tag)

        if cfg.save_grid_png and frames_by_step:
            ordered_idx = sorted(i for i in target_map.keys() if i in frames_by_step)
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
    if tag == "untrained":
        title = "DDPM CFG - Sample Trajectories (untrained)"
    else:
        title = "DDPM CFG - Sample Trajectories (trained)"
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
            ax.scatter(data[:, 0], data[:, 1], s=24, alpha=0.7, color="#4F8CC9", edgecolor="k", linewidth=0.5)
            ax.set_aspect("equal")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_xticks(np.arange(-4, 5, 2))
            ax.set_yticks(np.arange(-4, 5, 2))
            ax.grid(True, linestyle="--", color="gray", alpha=0.7, linewidth=1.0)
            t_val = t_vals[idx]
            ax.set_title(f"Step {idx} (t={t_val:.3f})", fontsize=12, pad=6)
            ax.tick_params(axis="both", which="major", labelsize=9, length=4)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.set_title("")
            ax.grid(False)

    plt.subplots_adjust(top=0.90, wspace=0.25, hspace=0.25)
    suffix = f"_{tag}" if tag else ""
    png_path = os.path.join(cfg.save_dir, f"diffusion_cfg_grid{suffix}.png")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_step_samples(frames_by_step, cfg, tag=""):
    base_dir = os.path.join(cfg.save_dir, "steps", tag or "trained")
    os.makedirs(base_dir, exist_ok=True)

    for step_idx, data in frames_by_step.items():
        npy_path = os.path.join(base_dir, f"step_{step_idx:04d}.npy")
        np.save(npy_path, data)

        if cfg.save_step_png:
            try:
                import matplotlib.pyplot as plt
            except Exception as exc:
                print(f"step plotting disabled {exc}")
                return

            plt.figure(figsize=(4, 4))
            plt.scatter(data[:, 0], data[:, 1], s=8, alpha=0.7, color="#4F8CC9")
            plt.axis("equal")
            plt.axis("off")
            png_path = os.path.join(base_dir, f"step_{step_idx:04d}.png")
            plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
            plt.close()


if __name__ == "__main__":
    cfg = Config()
    if cfg.sample_before_train:
        untrained = ConditionalNoiseNet(
            dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers, num_classes=cfg.num_classes
        ).to(cfg.device)
        original_dir = cfg.save_dir
        cfg.save_dir = cfg.save_dir_untrained
        sample(untrained, cfg, tag="untrained")
        cfg.save_dir = original_dir

    model = train(cfg)
    original_sample_class = cfg.sample_class
    sample(model, cfg, tag="trained_all")
    for class_id in range(cfg.num_classes):
        cfg.sample_class = class_id
        sample(model, cfg, tag=f"trained_c{class_id}")
    cfg.sample_class = original_sample_class
    print("done")
