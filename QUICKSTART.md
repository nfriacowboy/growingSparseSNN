# GrowingSparseSNN - Quick Start Guide 🚀

## ✅ What Was Created

### 1. **GitHub Repository**
- URL: https://github.com/nfriacowboy/growingSparseSNN
- Public repository with MIT license
- Initial commit with complete structure

### 2. **GrowingSparseSNN Architecture** (`src/models/growing_snn.py`)
- Dynamic SNN with LIF (Leaky Integrate-and-Fire) neurons
- **Neurogenesis**: Adds neurons when firing rate < 0.05
- **Pruning**: Removes neurons with firing rate < 0.005
- Starts with 64 neurons, grows up to 512-1024
- Optimized for AMD GPU with ROCm

### 3. **Test Environment** (`src/environments/grid_world.py`)
- 15×15 grid with agent and food
- Agent learns to forage (collect food)
- Observation: 2 channels (agent position, food positions)
- 4 actions: up, down, left, right

### 4. **Training System** (`src/training/trainer.py`)
- REINFORCE algorithm with baseline
- Automatic growth every 100 episodes
- Automatic pruning every 50 episodes
- Metrics exported to Prometheus

### 5. **Monitoring** (`src/monitoring/metrics.py`)
- Prometheus/OpenMetrics integrated
- Metrics: neuron count, firing rates, sparsity, rewards, energy
- Port: 8000
- Grafana dashboards via docker-compose

### 6. **Docker + ROCm** (`docker/`)
- Dockerfile based on `rocm/pytorch:rocm6.0`
- docker-compose.yml with SNN + Prometheus + Grafana
- Full support for AMD GPU

### 7. **Complete Tests** (`tests/`)
- `test_growth.py`: Tests neurogenesis
- `test_pruning.py`: Tests neuron pruning
- `test_learning.py`: Tests training and REINFORCE
- `test_environment.py`: Tests simulation environment
- Run with: `pytest tests/ -v --cov=src`

### 8. **Utility Scripts**
- `setup.sh`: Initial environment setup
- `run_tests.sh`: Runs tests with coverage
- `build_docker.sh`: Builds Docker image
- `demo.py`: Demo with visualizations

### 9. **Documentation**
- `README.md`: Main documentation
- `docs/architecture.md`: Detailed architecture
- `configs/training_config.yaml`: Hyperparameter configuration

## 🎯 Next Steps

### 1. Local Setup (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
./run_tests.sh

# Quick demo (trains 500 episodes)
python demo.py
```

### 2. Setup with Docker + ROCm (Recommended for AMD GPU)
```bash
# Build image
./build_docker.sh

# Or use docker-compose
cd docker
docker-compose up -d

# View logs
docker-compose logs -f snn-training

# Access container
docker exec -it growing-snn-train bash
```

### 3. Train Complete Model
```bash
# With default configuration
python src/training/train.py

# With custom configuration
python src/training/train.py --config configs/training_config.yaml

# With more episodes
python src/training/train.py --episodes 5000 --lr 0.001
```

### 4. Monitor Training
```bash
# Start monitoring services
cd docker
docker-compose up prometheus grafana

# Access:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - Raw metrics: http://localhost:8000/metrics
```

### 5. Suggested Experiments

#### Baseline: Fixed Network
```python
# Modify training_config.yaml:
training:
  grow_interval: 999999  # Never grows
  prune_interval: 999999  # Never prunes
```

#### Aggressive Growth
```python
training:
  grow_interval: 25       # Grows more frequently
  grow_threshold: 0.1     # Higher threshold
  neurons_per_grow: 64    # Adds more neurons
```

#### Aggressive Pruning
```python
training:
  prune_interval: 20
  prune_threshold: 0.01   # Removes less active neurons
```

## 📊 Implemented Metrics

| Metric | Description |
|--------|-------------|
| `snn_neuron_count` | Current number of neurons |
| `snn_avg_firing_rate` | Average firing rate |
| `snn_sparsity` | Proportion of inactive neurons |
| `snn_episode_reward` | Current episode reward |
| `snn_growth_events_total` | Total growth events |
| `snn_pruning_events_total` | Total pruning events |
| `snn_energy_estimate` | Energy estimate (spikes × neurons) |

## 🔬 Experimental Hypothesis

**H0**: A SNN with dynamic growth (64→512 neurons) + pruning learns better than a fixed 512-neuron network.

**Metrics to validate**:
1. **Sample efficiency**: Episodes until convergence
2. **Final performance**: Average reward after convergence
3. **Energy efficiency**: Total energy consumed
4. **Adaptation**: Performance on new tasks

## 🛠 Troubleshooting

### Issue: Norse not found
```bash
pip install norse
```

### Issue: ROCm not detected
```bash
# Check ROCm installation
rocm-smi

# Check PyTorch with ROCm
python -c "import torch; print(torch.__version__)"
# Should show something like: 2.1.0+rocm5.7

# If not, reinstall PyTorch for ROCm:
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

### Issue: GPU not detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, check drivers and ROCm
```

### Issue: Port 8000 in use
```bash
# Modify port in code or:
python src/training/train.py --metrics-port 8001
```

## 📚 File Structure

```
growingSparseSNN/
├── src/
│   ├── models/
│   │   └── growing_snn.py          # ⭐ Main model
│   ├── environments/
│   │   └── grid_world.py           # Simulation environment
│   ├── training/
│   │   ├── trainer.py              # ⭐ Training loop
│   │   └── train.py                # Main script
│   └── monitoring/
│       └── metrics.py              # Prometheus metrics
├── tests/                          # Unit tests
├── docker/                         # Docker + ROCm
├── configs/                        # YAML configurations
├── docs/                           # Documentation
├── demo.py                         # ⭐ Quick demo
└── README.md                       # Main documentation
```

## 🎓 Key Concepts

### Neurogenesis (Growth)
- Adds neurons when capacity is insufficient
- Trigger: avg_firing_rate < 0.05
- Preserves existing weights
- Initializes new ones with Kaiming + noise

### Pruning
- Removes inactive neurons
- Trigger: firing_rate < 0.005
- Maintains at least 32 neurons
- Reconstructs smaller network

### LIF Neurons
- Leaky Integrate-and-Fire
- τ_mem = 20ms, τ_syn = 10ms
- Threshold = 1.0
- Binary spikes (0 or 1)

### REINFORCE Learning
- Policy gradient with baseline
- Discount γ = 0.99
- Adam optimizer
- Gradient clipping (max_norm=1.0)

## 🚀 Project Status

✅ GitHub repository created  
✅ Architecture implemented  
✅ Complete tests  
✅ Docker + ROCm configured  
✅ Prometheus/Grafana monitoring  
✅ Functional demo  
✅ Complete documentation  

🔄 **Next**: Train and validate experimental hypothesis!

## 📞 Resources

- **Repository**: https://github.com/nfriacowboy/growingSparseSNN
- **Norse Docs**: https://norse.github.io/norse/
- **ROCm Docs**: https://rocm.docs.amd.com/
- **PyTorch SNN Tutorial**: https://snntorch.readthedocs.io/

---

**Created**: 2026-02-06  
**Author**: nfriacowboy  
**GPU Target**: AMD Radeon RX 6900 XT (ROCm 6.0)  
**Framework**: PyTorch + Norse + OpenMetrics
