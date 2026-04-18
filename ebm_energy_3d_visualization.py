import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from ebm_cd_mcmc import EnergyNet, Config, set_seed, sample_mog


class Arrow3D(FancyArrowPatch):
    """3D arrow patch for matplotlib."""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = [x, y, z]
        self._dxdydz = [dx, dy, dz]

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        
        xs = [x1, x1 + dx]
        ys = [y1, y1 + dy]
        zs = [z1, z1 + dz]
        
        xs, ys, zs = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        super().draw(renderer)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        
        xs = [x1, x1 + dx]
        ys = [y1, y1 + dy]
        zs = [z1, z1 + dz]
        
        xs, ys, zs = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        return np.min(zs)


def find_critical_points(X, Y, energy_grid, n_peaks=3, n_valleys=3):
    """
    Find local minima and maxima in the energy landscape.
    Returns: valley_coords, peak_coords
    """
    from scipy.signal import argrelextrema
    
    # Flatten and find indices
    flat_energy = energy_grid.flatten()
    
    # Find approximate minima and maxima using local extrema
    min_indices = argrelextrema(flat_energy, np.less, order=15)[0]
    max_indices = argrelextrema(flat_energy, np.greater, order=15)[0]
    
    # Get top valleys and peaks
    if len(min_indices) > 0:
        valley_energies = flat_energy[min_indices]
        valley_idx = min_indices[np.argsort(valley_energies)][:n_valleys]
    else:
        valley_idx = [np.argmin(flat_energy)]
    
    if len(max_indices) > 0:
        peak_energies = flat_energy[max_indices]
        peak_idx = max_indices[np.argsort(peak_energies)[-n_peaks:]]
    else:
        peak_idx = [np.argmax(flat_energy)]
    
    # Convert flat indices back to 2D
    valley_coords = np.unravel_index(valley_idx, energy_grid.shape)
    peak_coords = np.unravel_index(peak_idx, energy_grid.shape)
    
    valleys = list(zip(X[valley_coords], Y[valley_coords], energy_grid[valley_coords]))
    peaks = list(zip(X[peak_coords], Y[peak_coords], energy_grid[peak_coords]))
    
    return valleys, peaks


def add_arrows_to_plot(ax, X, Y, energy_grid, arrow_height=0.5, n_peaks=3, n_valleys=3):
    """
    Add arrows pointing to critical points (minima and maxima).
    """
    from scipy.signal import argrelextrema
    
    valleys, peaks = find_critical_points(X, Y, energy_grid, n_peaks, n_valleys)
    
    # Add green arrows pointing up from valleys (minima)
    for x, y, z in valleys:
        arrow = Arrow3D(x, y, z, 0, 0, arrow_height, 
                       mutation_scale=8, lw=1, arrowstyle='->', 
                       color='green', zorder=10)
        ax.add_artist(arrow)
    
    # Add red arrows pointing down from peaks (maxima)
    for x, y, z in peaks:
        arrow = Arrow3D(x, y, z, 0, 0, -arrow_height, 
                       mutation_scale=8, lw=1, arrowstyle='->', 
                       color='red', zorder=10)
        ax.add_artist(arrow)


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


def plot_3d_energy(model, cfg, tag="", real_data=None, save_dir=None):
    """
    Create a 3D surface plot of the energy landscape.
    """
    if save_dir is None:
        save_dir = cfg.save_dir
    
    print(f"Computing energy grid for 3D visualization...")
    X, Y, energy_grid = compute_energy_grid(
        model,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        grid_size=150,
        device=cfg.device
    )
    
    # Create figure with 3D projection
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, energy_grid, cmap='viridis', alpha=0.8, 
                           edgecolor='none', antialiased=True, rstride=2, cstride=2)
    
    # Add contour plot on the bottom
    ax.contourf(X, Y, energy_grid, levels=20, cmap='viridis', offset=energy_grid.min()-2, alpha=0.3)
    
    # Plot real data points as vertical lines if provided
    if real_data is not None:
        # Get energy values for real data points
        real_data_tensor = torch.FloatTensor(real_data).to(cfg.device)
        model.eval()
        with torch.no_grad():
            real_energies = model(real_data_tensor).cpu().numpy()
        
        # Plot real data as red points on the surface
        ax.scatter(real_data[:, 0], real_data[:, 1], real_energies, 
                  c='red', s=30, alpha=0.6, label='Real Data', depthshade=False)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('x', fontsize=11, fontweight='bold')
    ax.set_ylabel('y', fontsize=11, fontweight='bold')
    ax.set_zlabel('Energy', fontsize=11, fontweight='bold')
    
    title = f"Energy Landscape - {tag.capitalize()}" if tag else "Energy Landscape"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    png_path = os.path.join(save_dir, f"energy_landscape_3d{suffix}.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=120)
    print(f"Saved 3D energy plot to {png_path}")
    plt.close(fig)


def plot_comparison_3d(trained_model, untrained_model, cfg, real_data=None):
    """
    Create side-by-side 3D plots of trained vs untrained.
    """
    print(f"Computing energy grids for comparison...")
    X, Y, energy_trained = compute_energy_grid(
        trained_model,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        grid_size=150,
        device=cfg.device
    )
    
    X_u, Y_u, energy_untrained = compute_energy_grid(
        untrained_model,
        x_min=-10,
        x_max=10,
        y_min=-10,
        y_max=10,
        grid_size=150,
        device=cfg.device
    )
    
    # Create side-by-side subplots
    fig = plt.figure(figsize=(18, 7))
    
    # Trained model
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, energy_trained, cmap='viridis', alpha=0.8,
                            edgecolor='none', antialiased=True, rstride=2, cstride=2)
    
    if real_data is not None:
        real_data_tensor = torch.FloatTensor(real_data).to(cfg.device)
        trained_model.eval()
        with torch.no_grad():
            real_energies_t = trained_model(real_data_tensor).cpu().numpy()
        ax1.scatter(real_data[:, 0], real_data[:, 1], real_energies_t,
                   c='red', s=30, alpha=0.6, label='Real Data', depthshade=False)
    
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlabel('x', fontsize=10, fontweight='bold')
    ax1.set_ylabel('y', fontsize=10, fontweight='bold')
    ax1.set_zlabel('Energy', fontsize=10, fontweight='bold')
    ax1.set_title('Trained Model - Sculpted Landscape', fontsize=12, fontweight='bold', pad=15)
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Untrained model
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X_u, Y_u, energy_untrained, cmap='viridis', alpha=0.8,
                            edgecolor='none', antialiased=True, rstride=2, cstride=2)
    
    if real_data is not None:
        untrained_model.eval()
        with torch.no_grad():
            real_energies_u = untrained_model(real_data_tensor).cpu().numpy()
        ax2.scatter(real_data[:, 0], real_data[:, 1], real_energies_u,
                   c='red', s=30, alpha=0.6, label='Real Data', depthshade=False)
    
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlabel('x', fontsize=10, fontweight='bold')
    ax2.set_ylabel('y', fontsize=10, fontweight='bold')
    ax2.set_zlabel('Energy', fontsize=10, fontweight='bold')
    ax2.set_title('Untrained Model - Random Landscape', fontsize=12, fontweight='bold', pad=15)
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    fig.suptitle('EBM Energy Landscape Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    os.makedirs(cfg.save_dir, exist_ok=True)
    png_path = os.path.join(cfg.save_dir, "energy_landscape_3d_comparison.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=120)
    print(f"Saved comparison to {png_path}")
    plt.close(fig)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    
    # Sample real data
    real_data = sample_mog(500, cfg.device).cpu().numpy()
    print(f"Sampled {len(real_data)} real data points\n")
    
    # Load models
    print("Loading trained model...")
    trained_model = load_trained_model(cfg)
    
    print("Loading untrained model...")
    untrained_model = EnergyNet(dim=2, hidden=cfg.hidden, n_layers=cfg.n_layers).to(cfg.device)
    untrained_model.eval()
    
    # Plot individual 3D landscapes
    print("\nGenerating trained model 3D visualization...")
    plot_3d_energy(trained_model, cfg, tag="trained", real_data=real_data)
    
    print("Generating untrained model 3D visualization...")
    cfg_untrained = Config()
    cfg_untrained.save_dir = cfg_untrained.save_dir_untrained
    plot_3d_energy(untrained_model, cfg_untrained, tag="untrained", real_data=real_data, 
                   save_dir=cfg_untrained.save_dir)
    
    # Plot comparison
    print("\nGenerating side-by-side comparison...")
    plot_comparison_3d(trained_model, untrained_model, cfg, real_data=real_data)
    
    print("Done!")


if __name__ == "__main__":
    main()
