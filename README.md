# Generative Modelling: Comparative Study of Diffusion, Energy-Based, and Flow-Based Models

This repository contains implementations of contemporary generative modeling approaches: Energy-Based Models (EBM), Diffusion Probabilistic Models, Flow Matching, and Score-Based Models. All models are trained and evaluated on 2D synthetic datasets to facilitate theoretical understanding and visual interpretation.

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

### 2. EBM Energy Landscape Visualization
**File:** `ebm_energy_visualization.py`

Visualizes the learned energy landscape of trained and untrained EBM models.

**Key observations:**
- Trained models exhibit sharp energy minima at learned data modes
- Untrained models display flat, uniform energy landscapes
- Empirical validation of the theoretical relationship: $p(x) \propto \exp(-E(x))$

**Generate visualizations:**
```bash
python ebm_energy_visualization.py
```

**Outputs:**
- `outputs/ebm/trained/ebm_energy_grid_trained.png` - Trained energy landscape
- `outputs/ebm/untrained/ebm_energy_grid_untrained.png` - Untrained energy landscape

### 3. Diffusion Models (DDPM)
**File:** `diffusion_train.py`

Denoising diffusion probabilistic model (DDPM) implementation following Ho et al. (2020).

**Methodology:**
- Forward process: Progressive Gaussian noise addition to data
- Reverse process: Learned denoising through noise prediction network
- Training objective: Mean squared error on predicted noise
- Sampling: Stochastic reverse process with explicit noise injection

**Run training:**
```bash
python diffusion_train.py
```

**Key outputs:**
- `outputs/diffusion/trained/noise_net.pt` - Trained noise prediction network
- `outputs/diffusion/trained/diffusion_grid_trained.png` - Sampling progression visualization
- `outputs/diffusion/untrained/diffusion_grid_untrained.png` - Untrained baseline

### 4. Score-Based Generative Models
**File:** `diffusion_score_matching.py`

Score-based approach using score matching objective as an alternative to noise prediction.

**Methodology:**
- **Score function:** $\nabla_x \log p(x|t)$ representing log-probability gradients at different noise levels
- **Training objective:** Score matching loss
- **Sampling:** Reverse-time stochastic differential equation with learned scores
- **Theoretical advantage:** Direct gradient estimation versus indirect noise prediction

**Run training:**
```bash
python diffusion_score_matching.py
```

**Key outputs:**
- `outputs/diffusion_score/trained/score_net.pt` - Trained score network
- `outputs/diffusion_score/trained/diffusion_score_grid_trained.png` - Sampling progression
- `outputs/diffusion_score/untrained/diffusion_score_grid_untrained.png` - Untrained baseline

### 5. Flow Matching
**File:** `flow_matching_train.py`

Flow-based generative model using continuous-time matching between distributions.

**Methodology:**
- **Velocity network:** Learns the flow field from noise to data distribution
- **Training objective:** Conditional flow matching loss
- **Sampling:** Deterministic ODE integration from prior to data distribution
- **Advantage over diffusion:** Simplified training and deterministic generation process

**Run training:**
```bash
python flow_matching_train.py
```

**Key outputs:**
- `outputs/flow_matching/trained/velocity_net.pt` - Trained velocity network
- `outputs/flow_matching/trained/flow_matching_grid_trained.png` - Sampling progression
- `outputs/flow_matching/untrained/flow_matching_grid_untrained.png` - Untrained baseline

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
│   │   ├── noise_net.pt
│   │   └── diffusion_grid_trained.png
│   └── untrained/
│       └── diffusion_grid_untrained.png
├── diffusion_score/
│   ├── trained/
│   │   ├── score_net.pt
│   │   └── diffusion_score_grid_trained.png
│   └── untrained/
│       └── diffusion_score_grid_untrained.png
├── flow_matching/
│   ├── trained/
│   │   ├── velocity_net.pt
│   │   └── flow_matching_grid_trained.png
│   └── untrained/
│       └── flow_matching_grid_untrained.png
└── trajectories/
    └── (generated trajectory data)
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

## Visualization and Interpretation

### Sampling Progression Analysis

All models produce visualizations documenting the sampling trajectory across multiple time steps:

**Trained Models:**
- Progressive convergence toward the target data distribution (8-mode Gaussian mixture)
- Discernible structure in sample concentration patterns
- Final empirical distribution approaches training data distribution

**Untrained Models:**
- Baseline behavior without learned generative capacity
- Flow Matching: Deterministic but unstructured trajectories
- Diffusion/Score-Based: Dominated by stochastic noise with no coherent structure

### Behavioral Distinctions

**Deterministic Sampling (Flow Matching):**
- ODE-based generation without stochastic components
- Reproducible trajectories for given initial conditions
- Untrained models exhibit fixed, non-convergent patterns

**Stochastic Sampling (Diffusion and Score-Based):**
- Incorporates explicit Gaussian noise at each generation step
- Probabilistic reverse process following learned dynamics
- Untrained models demonstrate uniform noise dominance

### Energy-Based Model Characterization

**Trained energy landscape:**
- Sharp, concentrated minima corresponding to data modes
- Strong energy differentiation across sample space
- Validates relationship $p(x) \propto \exp(-E(x))$

**Untrained energy landscape:**
- Flat, structureless energy surface
- Absence of preferred sampling regions
- Confirms model has not learned data distribution