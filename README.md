# GrowingSparseSNN 🧠🌱

**Digital Organism with Dynamic Growing Spiking Neural Networks**

A research project exploring **neurogenesis** (neuron growth), **synaptic pruning**, and **continual learning** in Spiking Neural Networks (SNNs), optimized for AMD GPUs with ROCm.

## 🎯 Core Concepts

- **Dynamic Growth**: Neurons and connections emerge as the organism learns new patterns
- **Energy Efficiency**: Natural sparsity of SNNs + aggressive pruning + controlled growth
- **Continual Learning**: Learn without catastrophic forgetting
- **Hardware-Aware**: Optimized for AMD GPU (Gigabyte AI PRO R9700) using PyTorch + ROCm

## 🧪 Hypothesis

> A SNN with dynamically growing hidden layers (neurogenesis controlled by novelty/saturation metrics) + pruning based on low firing rates + learning via STDP or surrogate gradients, can learn exploration/foraging tasks with less energy and better adaptation than a fixed-size network of the same maximum capacity.

## 🏗️ Architecture

### GrowingSparseSNN Features

- **Input**: Environment observation (e.g., 15×15 grid × 2 channels → ~450 features)
- **Dynamic Hidden Layer**: Starts small (64 LIF neurons), grows up to max (512-1024)
- **Output**: 4 actions (up, down, left, right) - rate-based or temporal coding
- **Plasticity Mechanisms**:
  - **Growth (Neurogenesis)**: Add neurons when avg firing rate < threshold (0.05 spikes/timestep)
  - **Pruning**: Remove neurons with firing rate < 0.005 after evaluation window
  - **Synaptic Learning**: Surrogate gradient + REINFORCE or simple STDP

## 📦 Installation

### Prerequisites
- Docker + ROCm support (for GPU training, optional)
- AMD GPU with ROCm drivers (optional, can run on CPU)
- Python 3.10+
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

### Quick Start

#### Local Setup with UV
```bash
# Clone the repository
git clone https://github.com/nfriacowboy/growingSparseSNN.git
cd growingSparseSNN

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment (creates venv + installs deps)
./setup.sh

# Activate virtual environment
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Train model
python src/training/train.py
```

#### Using Docker + ROCm (alternative for GPU)
```bash
# Build Docker image with ROCm
docker build -t growing-snn:rocm -f docker/Dockerfile.rocm .

# Run container with GPU access
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd):/workspace growing-snn:rocm

# Inside container - run basic test
python tests/test_growth.py
```

## 🚀 Usage

### Basic Training

```python
from src.models.growing_snn import GrowingSparseSNN
from src.environments.grid_world import ForagingGrid

# Create organism
organism = GrowingSparseSNN(
    input_features=450, 
    init_hidden=64, 
    max_hidden=512
)

# Create environment
env = ForagingGrid(size=15, n_food=10)

# Train (with growth and pruning)
trainer = AdaptiveTrainer(organism, env)
trainer.train(episodes=1000, grow_interval=100, prune_interval=50)
```

## 📊 Monitoring

The project uses Prometheus + Grafana for real-time monitoring:

- Neuron count over time
- Firing rates and sparsity
- Energy consumption estimates
- Learning curves
- Growth/pruning events

Access dashboard at `http://localhost:3000` after starting services.

## 🧬 Project Structure

```
growingSparseSNN/
├── src/
│   ├── models/
│   │   ├── growing_snn.py      # Main GrowingSparseSNN class
│   │   ├── lif_neuron.py       # LIF neuron implementations
│   │   └── plasticity.py       # Growth/pruning/STDP rules
│   ├── environments/
│   │   ├── grid_world.py       # Foraging grid environment
│   │   └── base_env.py         # Base environment interface
│   ├── training/
│   │   ├── trainer.py          # Training loop with growth
│   │   └──rl_agent.py         # RL integration
│   └── monitoring/
│       ├── metrics.py          # OpenMetrics exporter
│       └── prometheus.py       # Prometheus client
├── tests/
│   ├── test_growth.py          # Growth mechanism tests
│   ├── test_pruning.py         # Pruning tests
│   └── test_learning.py        # Learning tests
├── docker/
│   ├── Dockerfile.rocm         # ROCm-enabled container
│   └── docker-compose.yml      # Services orchestration
├── configs/
│   ├── model_config.yaml       # Model hyperparameters
│   └── training_config.yaml    # Training settings
├── experiments/
│   └── notebooks/              # Jupyter notebooks for analysis
└── docs/
    └── architecture.md         # Detailed architecture docs
```

## 📈 Experiments

- Baseline: Fixed-size SNN (512 neurons)
- Growing: Dynamic growth from 64→512 neurons
- Growing+Pruning: Growth + aggressive pruning
- Metrics: Energy efficiency, sample efficiency, final performance

## 🔬 Based on Recent Research

- Structural plasticity in sparse SNNs (arXiv 2024/2025)
- Dynamic pruning + synaptic regeneration
- LIF neurons + STDP/R-STDP
- ROCm optimization patterns for SNN

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss proposed changes.

## 📧 Contact

- GitHub: [@nfriacowboy](https://github.com/nfriacowboy)
- Project: [growingSparseSNN](https://github.com/nfriacowboy/growingSparseSNN)

---

**Status**: 🚧 Active Development | **GPU**: AMD Radeon R9700 + ROCm | **Framework**: PyTorch + Norse
