import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ebm_cd_mcmc import EnergyNet, langevin_step, Config, set_seed


def load_trained_model(cfg):
    """Load the trained EBM model."""
    model = EnergyNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
    model_path = os.path.join(cfg.save_dir, "energy_net.pt")
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()
    return model


def compute_energy_grid(model, x_min=-10, x_max=10, y_min=-10, y_max=10, grid_size=100, device="cpu"):
    """
    Compute energy values on a 2D grid.
    Returns: grid_x, grid_y, energy_grid
    """
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid for batch evaluation
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points_tensor = torch.FloatTensor(points).to(device)
    
    model.eval()
    with torch.no_grad():
        energies = model(points_tensor).cpu().numpy()
    
    energy_grid = energies.reshape(X.shape)
    return X, Y, energy_grid


def compute_probability_from_energy(energy_grid, normalize=True):
    """Convert energy to unnormalized probability: p(x) ∝ exp(-E(x))"""
    prob_grid = np.exp(-energy_grid)
    if normalize:
        prob_grid = prob_grid / prob_grid.max()
    return prob_grid


def sample_and_visualize_energy_progression(model, cfg):
    """
    Generate samples from EBM and save energy visualization at each step.
    Returns list of energy grids and probability grids.
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(cfg.sample_count, 2, device=cfg.device)
    
    frames = []
    steps = cfg.cd_k * 4
    
    print(f"Generating {steps} steps of Langevin dynamics...")
    for i in range(steps):
        x = langevin_step(x, model, cfg.langevin_step_size, cfg.langevin_noise_scale)
        
        if i % cfg.save_every == 0 or i == steps - 1:
            # Compute energy grid
            X, Y, energy_grid = compute_energy_grid(
                model, 
                x_min=-10, 
                x_max=10, 
                y_min=-10, 
                y_max=10, 
                grid_size=100,
                device=cfg.device
            )
            prob_grid = compute_probability_from_energy(energy_grid, normalize=True)
            frames.append((X, Y, prob_grid))
            print(f"  Step {i+1}/{steps}")
    
    return frames


def save_energy_grid(frames, cfg, tag=""):
    """
    Save energy visualizations in a grid layout (similar to ebm_grid).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"grid plotting disabled {exc}")
        return
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    n_frames = len(frames)
    cols = max(1, cfg.grid_cols)
    rows = (n_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle("EBM Energy Visualization - Langevin MCMC Progression", fontsize=14, fontweight='bold')
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    total = cfg.cd_k * 4
    vmin, vmax = None, None
    
    # First pass: get global min/max for consistent colorbar
    all_probs = [frame[2] for frame in frames]
    vmin = min([p.min() for p in all_probs])
    vmax = max([p.max() for p in all_probs])
    
    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax.axis("off")
        
        if idx < n_frames:
            X, Y, prob_grid = frames[idx]
            im = ax.contourf(X, Y, prob_grid, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            step = min((idx * cfg.save_every) + 1, total)
            ax.set_title(f"step = {step}", fontsize=9, pad=2)
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('p(x) (unnormalized)', rotation=270, labelpad=20)
    
    suffix = f"_{tag}" if tag else ""
    png_path = os.path.join(cfg.save_dir, f"ebm_energy_grid{suffix}.png")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.1, dpi=120)
    print(f"Saved energy grid to {png_path}")
    plt.close(fig)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    
    # Generate for trained model
    print("\n" + "="*60)
    print("TRAINED MODEL")
    print("="*60)
    print("Loading trained EBM model...")
    model = load_trained_model(cfg)
    
    print("Generating energy visualization progression...")
    frames = sample_and_visualize_energy_progression(model, cfg)
    
    print(f"Creating grid with {len(frames)} frames...")
    save_energy_grid(frames, cfg, tag="trained")
    
    # Generate for untrained model
    print("\n" + "="*60)
    print("UNTRAINED MODEL")
    print("="*60)
    cfg_untrained = Config()
    cfg_untrained.save_dir = cfg_untrained.save_dir_untrained
    set_seed(cfg_untrained.seed)
    
    print("Loading untrained EBM model...")
    untrained_model = EnergyNet(dim=2, hidden=cfg_untrained.hidden, n_layers=cfg_untrained.n_layers).to(cfg_untrained.device)
    untrained_model.eval()
    
    print("Generating energy visualization progression...")
    frames_untrained = sample_and_visualize_energy_progression(untrained_model, cfg_untrained)
    
    print(f"Creating grid with {len(frames_untrained)} frames...")
    save_energy_grid(frames_untrained, cfg_untrained, tag="untrained")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
