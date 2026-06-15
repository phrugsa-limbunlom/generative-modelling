"""Render the DDPM-CFG reverse-diffusion process as an animated GIF.

Replays the trained model's denoising trajectory (noise -> 8-mode mixture) and
lays the panels out in the same 3x3 grid as diffusion_cfg_grid_combined.png:
untrained, trained class 0-6, trained all-classes.
"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from diffusion_train_cfg import (
    Config,
    ConditionalNoiseNet,
    make_schedule,
    p_sample_step,
    predict_eps,
    set_seed,
)

OUT_PATH = "outputs/diffusion_cfg/diffusion_cfg_reverse.gif"
N_FRAMES = 60          # frames sampled from the 201-step trajectory
HOLD_FRAMES = 12       # extra frames holding the final (converged) state
FPS = 12

ACCENT_TRAINED = "#4F8CC9"
ACCENT_ALL = "#107A60"
ACCENT_UNTRAINED = "#CB6060"


def sample_trajectory(model, cfg, sample_class):
    """Run reverse diffusion, returning a list of (t, points) for every step."""
    model.eval()
    betas, alphas, alpha_bars = make_schedule(cfg, cfg.device)
    frames = []
    with torch.no_grad():
        x = torch.randn(cfg.sample_count, 2, device=cfg.device)
        if sample_class < 0:
            labels = torch.randint(0, cfg.num_classes, (x.size(0),), device=cfg.device)
        else:
            labels = torch.full((x.size(0),), sample_class, device=cfg.device, dtype=torch.long)

        frames.append((1.0, x.cpu().numpy().copy()))
        step = 0
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
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
            step += 1
            t_val = (cfg.t_steps - step) / cfg.t_steps
            frames.append((t_val, x.cpu().numpy().copy()))
    return frames


def select_indices(n_total, n_frames):
    idx = np.linspace(0, n_total - 1, n_frames).round().astype(int)
    return list(dict.fromkeys(idx.tolist()))  # unique, preserve order


def main():
    cfg = Config()
    cfg.use_cfg = True

    # untrained network (matches the red "Untrained" panel)
    set_seed(cfg.seed)
    untrained = ConditionalNoiseNet(
        dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers, num_classes=cfg.num_classes
    ).to(cfg.device)

    # trained network
    weights = os.path.join(cfg.save_dir, "noise_net_cfg.pt")
    trained = ConditionalNoiseNet(
        dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers, num_classes=cfg.num_classes
    ).to(cfg.device)
    trained.load_state_dict(torch.load(weights, map_location=cfg.device))

    # panel order matches the static grid: untrained, c0..c6, all
    panels = [("Untrained", ACCENT_UNTRAINED, untrained, 0)]
    for class_id in range(7):
        panels.append((f"Trained - class {class_id}", ACCENT_TRAINED, trained, class_id))
    panels.append(("Trained - all classes", ACCENT_ALL, trained, -1))

    print("sampling trajectories...")
    set_seed(cfg.seed)
    traj = []
    for title, _, model, cls in panels:
        traj.append(sample_trajectory(model, cfg, cls))
        print(f"  done: {title}")

    total = len(traj[0])
    sel = select_indices(total, N_FRAMES)
    sel = sel + [sel[-1]] * HOLD_FRAMES  # hold on the converged frame

    # figure / axes
    fig, axes = plt.subplots(3, 3, figsize=(11, 11.6))
    fig.suptitle("DDPM with Classifier-Free Guidance — reverse diffusion",
                 fontsize=18, fontweight="bold", y=0.975)
    progress = fig.text(0.5, 0.935, "", ha="center", fontsize=12, color="#475569")

    scatters = []
    for ax, (title, accent, _, _) in zip(axes.ravel(), panels):
        first = traj[len(scatters)][sel[0]][1]
        sc = ax.scatter(first[:, 0], first[:, 1], s=9, alpha=0.65,
                        color=accent, linewidth=0)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(-4, 5, 2))
        ax.set_yticks(np.arange(-4, 5, 2))
        ax.grid(True, linestyle="--", color="gray", alpha=0.5, linewidth=0.8)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=11, color=accent, fontweight="bold", pad=4)
        scatters.append(sc)

    fig.tight_layout(rect=[0, 0, 1, 0.925])

    def update(frame_i):
        idx = sel[frame_i]
        t_val = traj[-1][idx][0]
        progress.set_text(f"t = {t_val:.3f}    (noise → data)")
        for sc, frames in zip(scatters, traj):
            sc.set_offsets(frames[idx][1])
        return scatters + [progress]

    print(f"rendering {len(sel)} frames -> {OUT_PATH}")
    anim = FuncAnimation(fig, update, frames=len(sel), interval=1000 / FPS, blit=False)
    anim.save(OUT_PATH, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
