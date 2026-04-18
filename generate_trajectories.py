"""
Unified Trajectory Visualization for Three Generative Models
Generates clean visualizations for:
- Diffusion Model (DDPM)
- Flow Matching
- Score-Based Diffusion

Single file that produces all trajectories cleanly.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

# Import models
from diffusion_train import NoiseNet as DiffusionNoiseNet, Config as DiffusionConfig, make_schedule as diffusion_make_schedule, set_seed
from flow_matching_train import VelocityNet, Config as FlowConfig, sample_base
from diffusion_score_matching import ScoreNet, Config as ScoreConfig, make_schedule as score_make_schedule


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "outputs/trajectories"
PARTICLE_COUNT = 100
SHOW_PARTICLES = 15  # Particles to highlight in trajectory plot
ANIMATION_PARTICLES = 50  # Particles in animation

# Color schemes
COLORS = {
    "diffusion": "#1f77b4",      # Blue
    "flow": "#2ca02c",           # Green
    "score": "#ff7f0e"           # Orange
}

TARGET_CENTERS = np.array([
    [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0],
    [2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]
])


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_diffusion_model(cfg):
    """Load trained diffusion model."""
    model = DiffusionNoiseNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
    model_path = os.path.join(cfg.save_dir, "noise_net.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        model.eval()
        return model
    return None


def load_flow_model(cfg):
    """Load trained flow matching model."""
    model = VelocityNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
    model_path = os.path.join(cfg.save_dir, "velocity_net.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        model.eval()
        return model
    return None


def load_score_model(cfg):
    """Load trained score-based diffusion model."""
    model = ScoreNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
    model_path = os.path.join(cfg.save_dir, "score_net.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        model.eval()
        return model
    return None


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def generate_diffusion_trajectory(model, cfg, n_samples):
    """Generate trajectory for diffusion model."""
    betas, alphas, alpha_bars = diffusion_make_schedule(cfg, cfg.device)
    trajectories = []
    t_values = []
    
    with torch.no_grad():
        x = torch.randn(n_samples, 2, device=cfg.device)
        trajectories.append(x.cpu().numpy().copy())
        t_values.append(0.0)
        
        for t_idx in reversed(range(cfg.t_steps)):
            t = torch.full((x.size(0),), (t_idx + 1) / cfg.t_steps, device=cfg.device)
            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]
            alpha_bar_t = alpha_bars[t_idx]
            alpha_bar_prev = alpha_bars[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=x.device)
            
            eps = model(x, t)
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            mean = coef1 * (x - coef2 * eps)
            
            if t_idx > 0:
                var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
            
            if (cfg.t_steps - t_idx - 1) % cfg.save_every == 0 or t_idx == 0:
                trajectories.append(x.cpu().numpy().copy())
                t_values.append(t.cpu()[0].item())
    
    return trajectories, t_values


def generate_flow_trajectory(model, cfg, n_samples):
    """Generate trajectory for flow matching model."""
    trajectories = []
    t_values = []
    
    with torch.no_grad():
        x = sample_base(n_samples, cfg.device)
        trajectories.append(x.cpu().numpy().copy())
        t_values.append(0.0)
        
        t_grid = torch.linspace(0.0, 1.0, cfg.ode_steps + 1, device=cfg.device)
        dt = 1.0 / cfg.ode_steps
        
        for i in range(cfg.ode_steps):
            t = t_grid[i].expand(x.size(0))
            k1 = model(x, t)
            k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = model(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
            if (i + 1) % cfg.save_every == 0 or i == cfg.ode_steps - 1:
                trajectories.append(x.cpu().numpy().copy())
                t_values.append(t_grid[i + 1].item())
    
    return trajectories, t_values


def generate_score_trajectory(model, cfg, n_samples):
    """Generate trajectory for score-based diffusion model."""
    betas, alphas, alpha_bars = score_make_schedule(cfg, cfg.device)
    trajectories = []
    t_values = []
    save_every = getattr(cfg, 'save_every', 20)
    
    with torch.no_grad():
        x = torch.randn(n_samples, 2, device=cfg.device)
        trajectories.append(x.cpu().numpy().copy())
        t_values.append(0.0)
        
        step_count = 0
        for t_idx in reversed(range(cfg.t_steps)):
            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]
            alpha_bar_t = alpha_bars[t_idx]
            alpha_bar_prev = alpha_bars[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=x.device)
            t = torch.full((x.size(0),), (t_idx + 1) / len(betas), device=x.device)
            
            score = model(x, t)
            coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x + coef * score)
            
            if t_idx > 0:
                var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
            
            step_count += 1
            if step_count % save_every == 0 or t_idx == 0:
                trajectories.append(x.cpu().numpy().copy())
                t_values.append(t.cpu()[0].item())
    
    return trajectories, t_values


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_particle_trajectories(trajectories, name, color):
    """Plot individual particle trajectories with legend in upper left."""
    n_samples = trajectories[0].shape[0]
    np.random.seed(42)
    indices = np.random.choice(n_samples, min(SHOW_PARTICLES, n_samples), replace=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create color map
    cmap = plt.colormaps['tab20']
    colors_particles = cmap(np.linspace(0, 1, len(indices)))
    
    # Plot target distribution
    ax.scatter(TARGET_CENTERS[:, 0], TARGET_CENTERS[:, 1], s=120, c='red', 
               marker='x', linewidths=2.5, label='Target Distribution', zorder=10)
    
    # Plot particle trajectories
    for idx_num, (particle_idx, p_color) in enumerate(zip(indices, colors_particles)):
        positions = np.array([traj[particle_idx] for traj in trajectories])
        
        # Draw trajectory line
        ax.plot(positions[:, 0], positions[:, 1], '-', alpha=0.5, color=p_color, linewidth=1.5, zorder=2)
        
        # Draw start point
        ax.scatter(positions[0, 0], positions[0, 1], s=60, color=p_color, 
                  marker='o', edgecolors='black', linewidths=1, zorder=4)
        
        # Draw end point
        ax.scatter(positions[-1, 0], positions[-1, 1], s=100, color=p_color, 
                  marker='*', edgecolors='black', linewidths=1, zorder=4)
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title(f'{name} - Individual Particle Trajectories (trained)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create legend items
    legend_elements = []
    
    # Add target distribution
    from matplotlib.lines import Line2D
    legend_elements.append(Line2D([0], [0], marker='x', color='red', markeredgecolor='red',
                                  markersize=10, markeredgewidth=2.5, linestyle='None',
                                  label='Target Distribution'))
    
    # Add particle labels with colors
    for particle_idx, p_color in zip(indices, colors_particles):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=p_color,
                                     markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                                     label=f'Particle {particle_idx}'))
    
    # Add legend in upper left corner with 2 columns
    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                      frameon=True, fancybox=False, shadow=False, framealpha=0.95,
                      borderpad=0.8, labelspacing=0.6, handlelength=1.5, handletextpad=0.5,
                      ncol=2, columnspacing=0.8)
    legend.get_frame().set_linewidth(1.0)
    
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_particle_trajectories.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_vector_field(model, cfg, name):
    """Plot vector field (noise/velocity/score) for all generative models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Determine field type for title
    if 'Score' in name:
        field_type = 'Score Fields'
    elif 'Flow' in name:
        field_type = 'Velocity Fields'
    else:  # Diffusion Model
        field_type = 'Noise Fields'
    
    fig.suptitle(f'{name} - Learned {field_type} (trained)', fontsize=14, fontweight='bold')
    
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    x = np.linspace(x_min, x_max, 15)
    y = np.linspace(y_min, y_max, 15)
    X, Y = np.meshgrid(x, y)
    
    t_samples = [0.1, 0.5, 0.9] if 'Score' in name else [0.0, 0.5, 1.0]
    
    model.eval()
    for ax_idx, t_val in enumerate(t_samples):
        ax = axes[ax_idx]
        
        points = np.stack([X.flatten(), Y.flatten()], axis=1)
        points_tensor = torch.FloatTensor(points).to(cfg.device)
        t_tensor = torch.full((points.shape[0],), t_val, device=cfg.device)
        
        with torch.no_grad():
            vectors = model(points_tensor, t_tensor).cpu().numpy()
        
        V_x = vectors[:, 0].reshape(X.shape)
        V_y = vectors[:, 1].reshape(Y.shape)
        
        # Compute magnitude for color mapping
        V_mag = np.sqrt(V_x**2 + V_y**2)
        V_mag_max = np.max(V_mag)
        if V_mag_max > 0:
            V_mag_normalized = V_mag / V_mag_max
        else:
            V_mag_normalized = V_mag
        
        # Normalize vectors for consistent arrow length
        V_mag_safe = np.sqrt(V_x**2 + V_y**2)
        V_mag_safe[V_mag_safe == 0] = 1
        V_x_norm = V_x / V_mag_safe
        V_y_norm = V_y / V_mag_safe
        
        # Create quiver plot with magnitude colormapping
        # Using 'inferno' colormap for blue (dark) to yellow (bright) gradient
        q = ax.quiver(X, Y, V_x_norm, V_y_norm, V_mag, 
                     cmap='inferno', scale=30, scale_units='width', width=0.0035,
                     edgecolors='none', alpha=0.9)
        
        # Determine magnitude label based on model type
        if 'Score' in name:
            magnitude_label = 'Score Magnitude'
        elif 'Flow' in name:
            magnitude_label = 'Velocity Magnitude'
        else:  # DDPM and other diffusion models predict noise
            magnitude_label = 'Noise Magnitude'
        
        # Add colorbar for this subplot
        cbar = plt.colorbar(q, ax=ax, pad=0.02, shrink=0.95)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(magnitude_label, fontsize=10)
        
        # Target markers (if they exist)
        if hasattr(cfg, 'device') and 'TARGET_CENTERS' in globals():
            ax.scatter(TARGET_CENTERS[:, 0], TARGET_CENTERS[:, 1], 
                      s=50, c='red', marker='x', linewidths=1.5, zorder=10)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f't = {t_val:.1f}', fontsize=12, fontweight='bold')
        
        # Clean styling - remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{name.lower().replace(' ', '_')}_field.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    set_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load configurations
    diff_cfg = DiffusionConfig()
    flow_cfg = FlowConfig()
    score_cfg = ScoreConfig()
    
    # Load models
    print("Loading models...")
    diff_model = load_diffusion_model(diff_cfg)
    flow_model = load_flow_model(flow_cfg)
    score_model = load_score_model(score_cfg)
    
    # Generate visualizations for each model
    models = [
        ("Diffusion Model", diff_model, diff_cfg, generate_diffusion_trajectory, COLORS["diffusion"]),
        ("Flow Matching", flow_model, flow_cfg, generate_flow_trajectory, COLORS["flow"]),
        ("Score-Based Diffusion", score_model, score_cfg, generate_score_trajectory, COLORS["score"]),
    ]
    
    for name, model, cfg, gen_func, color in models:
        trajectories, t_values = gen_func(model, cfg, PARTICLE_COUNT)
    
        plot_particle_trajectories(trajectories, name, color)
        
        plot_vector_field(model, cfg, name)

    
if __name__ == "__main__":
    main()