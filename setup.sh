#!/bin/bash
# Quick setup script for GrowingSparseSNN

set -e

echo "🧠 Setting up GrowingSparseSNN..."

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints results logs

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"
else
    echo "⚠ Not running in Docker. Consider using docker-compose for full setup."
fi

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Check PyTorch
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch found: $TORCH_VERSION"
    
    # Check for CUDA/ROCm
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        echo "✓ GPU detected: $GPU_NAME"
        
        if [[ $TORCH_VERSION == *"rocm"* ]]; then
            echo "✓ ROCm support detected!"
        fi
    else
        echo "⚠ No GPU detected. Will run on CPU."
    fi
else
    echo "✗ PyTorch not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check norse
if python3 -c "import norse" 2>/dev/null; then
    echo "✓ Norse (SNN library) found"
else
    echo "⚠ Norse not found. Installing..."
    pip install norse
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Quick start:"
echo "  1. Run tests: pytest tests/ -v"
echo "  2. Train model: python src/training/train.py"
echo "  3. View metrics: docker-compose up prometheus grafana"
echo ""
echo "For full setup with monitoring:"
echo "  docker-compose up"
echo ""
