# GrowingSparseSNN Architecture

## Overview

GrowingSparseSNN implements a biologically-inspired neural network that can dynamically grow (add neurons) and prune (remove neurons) during learning. This document describes the architecture in detail.

## Network Structure

### Input Layer
- Receives flattened grid observations from the environment
- Default: 450 features (15×15 grid × 2 channels)
- Channel 0: Agent position
- Channel 1: Food locations

### Hidden Layer (LIF Neurons)
- **Start Size**: 64 neurons (configurable)
- **Max Size**: 512-1024 neurons (configurable)
- **Neuron Type**: Leaky Integrate-and-Fire (LIF)
- **Dynamics**: 
  - Membrane time constant (τ_mem): 20ms
  - Synaptic time constant (τ_syn): 10ms
  - Threshold voltage: 1.0

### Output Layer
- 4 neurons (one per action)
- Represents action values in rate-coding scheme
- Actions: {0: up, 1: down, 2: left, 3: right}

## Neurogenesis (Growth)

### Trigger Conditions
Growth is triggered when:
1. Average firing rate < threshold (default: 0.05 spikes/timestep)
2. Current size < max_hidden
3. Growth interval reached (default: every 100 episodes)

### Growth Process
1. **Determine Growth Size**: Add N neurons (default: 32)
2. **Expand Input Weights**: 
   - Keep existing neuron weights intact
   - Initialize new neuron weights with Kaiming normal
   - Add small noise for diversity
3. **Expand Output Weights**:
   - Keep existing connections
   - Initialize new connections with Kaiming normal
4. **Reset State**: Clear LIF state and firing history for new neurons

### Rationale
Low average firing rates suggest the network lacks capacity to represent the current task complexity. Adding neurons provides new representational capacity.

## Pruning

### Trigger Conditions
Pruning is triggered when:
1. Sparsity > threshold (default: 30% of neurons with rate < 0.005)
2. Pruning interval reached (default: every 50 episodes)
3. Current size > min_neurons (safety: keep at least 32)

### Pruning Process
1. **Identify Low-Activity Neurons**: firing_rate < threshold (0.005)
2. **Safety Check**: Keep at least min_neurons most active neurons
3. **Remove Neurons**: Delete weights for low-activity neurons
4. **Reconstruct Network**: Create new layers with reduced size
5. **Reset State**: Clear LIF state

### Rationale
Neurons with consistently low firing rates are not contributing to learning and consume unnecessary computation and memory. Removing them improves efficiency.

## Learning Algorithm

### REINFORCE Policy Gradient
We use the REINFORCE algorithm with baseline for policy optimization:

1. **Collect Episode**: Run episode, store (state, action, reward) tuples
2. **Compute Returns**: Calculate discounted returns with γ=0.99
3. **Baseline**: Maintain exponential moving average of returns for variance reduction
4. **Policy Loss**: 
   ```
   L = -Σ log π(a|s) × (R - baseline)
   ```
5. **Optimize**: Adam optimizer with gradient clipping (max_norm=1.0)

### Key Features
- **Surrogate Gradients**: SNN spikes are non-differentiable; we use surrogate gradients from Norse
- **Rate Coding**: Output layer uses rate-based interpretation (spike count / timesteps)
- **Variance Reduction**: Baseline subtraction reduces gradient variance

## Temporal Dynamics

### Spiking Simulation
Each forward pass simulates T timesteps (default: 10):

For t = 1 to T:
1. Input → Hidden: Linear transform + LIF dynamics
2. Generate spikes: Binary events when membrane voltage > threshold
3. Hidden → Output: Linear transform
4. Accumulate output spikes

Output = total_spikes / T (rate coding)

### Why Spiking?
- **Energy Efficiency**: Sparse spikes → fewer computations
- **Temporal Information**: Spike timing can encode information
- **Biological Plausibility**: Closer to real neurons
- **Natural Sparsity**: No need for artificial regularization

## Growth/Pruning Strategy

### Adaptive Growth Heuristic
```
if avg_firing_rate < 0.05:
    # Network is underutilized or saturated
    if performance_plateau:
        GROW()  # Add capacity
    else:
        PRUNE()  # Remove dead neurons
```

### Sparsity-Based Pruning
```
sparsity = fraction of neurons with rate < 0.005
if sparsity > 0.3:
    PRUNE()  # Many dead neurons, clean up
```

## Monitoring and Metrics

### Tracked Metrics
- **Structure**: neuron count, growth/pruning events
- **Activity**: firing rates, sparsity
- **Learning**: episode rewards, losses
- **Energy**: estimated as spikes × neurons

### Prometheus Export
Metrics exposed at `http://localhost:8000/metrics`:
- `snn_neuron_count`
- `snn_avg_firing_rate`
- `snn_sparsity`
- `snn_episode_reward`
- `snn_growth_events_total`
- `snn_pruning_events_total`

### Grafana Dashboards
Visualize:
- Network size over time
- Learning curves
- Activity heatmaps
- Energy consumption

## Hardware Optimization

### GPU Acceleration (ROCm)
- **ROCm Stack**: PyTorch compiled with ROCm support
- **AMD GPU**: Optimized for RDNA2 architecture (RX 6000 series)
- **Tensor Operations**: All computations on GPU
- **Batch Processing**: Small batch size (8-32) for simulation speed

### Memory Efficiency
- **Sparse Patterns**: Many neurons have zero firing → natural sparsity
- **Dynamic Sizing**: Only allocate memory for active neurons
- **State Management**: Reset states between episodes to prevent memory leaks

### Compute Patterns
- **Avoid Python Loops**: All timestep simulation in vectorized ops
- **Fused Operations**: Use PyTorch's autograd for efficient backprop
- **Mixed Precision**: Can use FP16 for inference (not critical for small networks)

## Future Extensions

### Synaptic Plasticity
- **STDP**: Spike-Timing-Dependent Plasticity for unsupervised learning
- **Homeostasis**: Self-regulating firing rates
- **Heterosynaptic Plasticity**: Competition between synapses

### Structural Plasticity
- **Synapse-Level Growth**: Add individual connections, not just neurons
- **Dendritic Spines**: Model synaptic turnover explicitly
- **Activity-Dependent Rules**: More sophisticated growth triggers

### Continual Learning
- **Task Incremental**: Learn new tasks without forgetting
- **Elastic Weight Consolidation**: Protect important weights
- **Progressive Networks**: Dedicate neurons to specific tasks

## References

- Norse: https://github.com/norse/norse
- LIF Neurons: Gerstner & Kistler (2002)
- REINFORCE: Williams (1992)
- ROCm: https://rocm.docs.amd.com/

---

**Last Updated**: 2026-02-06  
**Version**: 0.1.0
