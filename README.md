# Generative Modelling - Energy-Based and Flow-Based Models

Implementations of various generative models including Energy-Based Models (EBM), Diffusion Models, Flow Matching, and Score Matching on 2D toy datasets.

## Models Implemented

### 1. Energy-Based Models (EBM) with Contrastive Divergence
**File:** `ebm_cd_mcmc.py`

Trains an energy network using Contrastive Divergence (CD) with Langevin MCMC sampling.

- **Training objective:** Minimize energy of real data, maximize energy of model samples
- **Sampling:** Langevin dynamics for both training and inference
- **Architecture:** Multi-layer perceptron with SiLU activations
- **Data:** 2D mixture of 8 Gaussians

**Run training:**
```bash
python ebm_cd_mcmc.py
```

**Key outputs:**
- `outputs/ebm/trained/energy_net.pt` - Trained energy network
- `outputs/ebm/trained/ebm_grid_trained.png` - Sampling progression grid
- `outputs/ebm/untrained/` - Untrained model samples for comparison

### 2. EBM Energy Visualization
**File:** `ebm_energy_visualization.py`

Visualizes the learned energy landscape of trained and untrained EBM models.

- **Trained model:** Shows sharp energy minima at learned data modes
- **Untrained model:** Shows flat, uniform energy landscape
- **Visualization:** 13-frame grid showing MCMC progression with energy heatmaps
- **Color scheme:** viridis (purple=high energy/low probability, yellow=low energy/high probability)

**Generate visualizations:**
```bash
python ebm_energy_visualization.py
```

**Outputs:**
- `outputs/ebm/trained/ebm_energy_grid_trained.png` - Trained energy landscape
- `outputs/ebm/untrained/ebm_energy_grid_untrained.png` - Untrained energy landscape

**Key insight:** The relationship is $p(x) \propto \exp(-E(x))$
- Trained model: Real data at low energy → high probability ✓
- Untrained model: All points at random energy → uniform probability ✗

### 3. Diffusion Models
**File:** `diffusion_train.py`

Denoising diffusion probabilistic model (DDPM) implementation.

- Forward process: Gradually add Gaussian noise to data
- Reverse process: Learn to denoise back to data
- **Outputs:** `outputs/diffusion/trained/noise_net.pt`

**Run training:**
```bash
python diffusion_train.py
```

### 4. Score Matching / Diffusion Score Models
**File:** `diffusion_score_matching.py`

Score-based generative models using score matching objective.

- **Score function:** Learns $\nabla_x \log p(x|t)$ for different noise levels
- **Sampling:** Reverse-time SDE with learned scores
- **Outputs:** `outputs/diffusion_score/trained/score_net.pt`

**Run training:**
```bash
python diffusion_score_matching.py
```

### 5. Flow Matching
**File:** `flow_matching_train.py`

Flow-based generative model matching data and noise distributions through continuous time.

- **Velocity network:** Learns flow from noise to data
- **Training:** Conditional flow matching objective
- **Sampling:** ODE integration from noise
- **Outputs:** `outputs/flow_matching/trained/velocity_net.pt`

**Run training:**
```bash
python flow_matching_train.py
```

## Directory Structure

```
outputs/
├── ebm/
│   ├── trained/
│   │   ├── energy_net.pt
│   │   ├── ebm_grid_trained.png
│   │   └── ebm_energy_grid_trained.png
│   └── untrained/
│       ├── ebm_samples_untrained.npy
│       └── ebm_energy_grid_untrained.png
├── diffusion/
│   ├── trained/
│   │   └── noise_net.pt
│   └── untrained/
├── diffusion_score/
│   ├── trained/
│   │   └── score_net.pt
│   └── untrained/
├── ebm/
│   └── energy_net.pt
└── flow_matching/
    ├── trained/
    │   └── velocity_net.pt
    └── untrained/
```

## Requirements

- PyTorch
- NumPy
- Matplotlib
- scikit-learn (optional, for k-means in EBM visualization)

## Key Concepts

### Energy-Based Models
- **Energy function:** Maps data points to scalar energy values
- **Training:** Contrastive divergence - push down energy of real data, push up energy of generated samples
- **Sampling:** Langevin dynamics (gradient descent with noise)
- **Mathematical:** $p(x) = \frac{1}{Z} \exp(-E(x))$

### Diffusion Models
- **Noise scheduling:** Progressive noise addition over time steps
- **Reverse process:** Learn to denoise gradually
- **Score matching:** Directly learn gradients of log-probability

### Flow Matching
- **Continuous dynamics:** ODE-based generation
- **Velocity network:** Direct learning of sample paths
- **Advantage:** Simpler training than diffusion, single ODE solver

## Visualization Interpretation

### Trained vs Untrained EBM Energy

**Trained Model:**
- Shows sharp, concentrated energy minima (bright yellow peaks)
- Multiple modes corresponding to data distribution
- Clear structure: low energy = high probability at data regions

**Untrained Model:**
- Flat, uniform energy landscape
- No learned structure
- No preferred regions for sampling