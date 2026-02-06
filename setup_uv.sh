#!/bin/bash
# Setup GrowingSparseSNN with UV (fast Python package manager)
# UV is much faster than pip and includes venv management
# Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh

set -e

echo "🧠 Setting up GrowingSparseSNN with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "⚠ UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✓ UV installed successfully"
fi

UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
echo "✓ UV found: $UV_VERSION"

# Create virtual environment with UV
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with UV..."
    uv venv .venv --python 3.10
    echo "✓ Virtual environment created at .venv/"
else
    echo "✓ Virtual environment already exists at .venv/"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with UV (much faster than pip)
echo "Installing dependencies with UV..."
uv pip install -r requirements.txt

echo ""
echo "✅ Setup complete with UV!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p checkpoints results logs

# Check PyTorch installation
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch found: $TORCH_VERSION"
    
    # Check for CUDA/ROCm
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        echo "✓ GPU detected: $GPU_NAME"
        
        if [[ $TORCH_VERSION == *"rocm"* ]]; then
            echo "✓ ROCm support detected!"
        fi
    else
        echo "⚠ No GPU detected. Will run on CPU."
    fi
else
    echo "⚠ PyTorch not found. You may need to install it manually:"
    echo "  uv pip install torch --index-url https://download.pytorch.org/whl/rocm5.7"
fi

# Check Norse
if python -c "import norse" 2>/dev/null; then
    echo "✓ Norse (SNN library) found"
else
    echo "⚠ Norse not found. Installing..."
    uv pip install norse
fi

echo ""
echo "🚀 Quick start:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Run tests: pytest tests/ -v"
echo "  3. Train model: python src/training/train.py"
echo "  4. Run demo: python demo.py"
echo ""
echo "For ROCm PyTorch (AMD GPU):"
echo "  uv pip install torch --index-url https://download.pytorch.org/whl/rocm5.7"
echo ""
