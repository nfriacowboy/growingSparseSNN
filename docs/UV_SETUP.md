# UV Package Manager

This project uses [UV](https://github.com/astral-sh/uv) exclusively for Python package management.

## Why UV Only?

- **10-100x faster** than pip
- **Reliable**: Resolves dependencies correctly every time
- **Unified tool**: Handles venv creation, package installation, and dependency resolution
- **Modern**: Built with Rust for maximum performance
- **No pip needed**: UV is a complete replacement

## Installation

```bash
# Install UV (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

## Quick Start

```bash
# One command to setup everything
./setup.sh

# Activate the virtual environment
source .venv/bin/activate
```

### Manual Setup
```bash
# Create virtual environment
uv venv .venv --python 3.10

# Activate it
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies (much faster than pip!)
uv pip install -r requirements.txt

# Install specific packages
uv pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
uv pip install norse
```

### Managing Dependencies

```bash
# Install new package
uv pip install <package>

# Install with specific version
uv pip install torch==2.1.0

# Upgrade package
uv pip install --upgrade <package>

# List installed packages
uv pip list

# Freeze dependencies
uv pip freeze > requirements.txt

# Uninstall package
uv pip uninstall <package>
```

### Running Commands in Virtual Environment

```bash
# Activate venv first
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Train model
python src/training/train.py

# Run demo
python demo.py

# Deactivate when done
deactivate
```

## UV vs pip Performance

| Command | pip | UV | Speedup |
|---------|-----|-----|---------|
| Install requirements.txt | ~45s | ~2s | **22x faster** |
| Resolve dependencies | ~30s | ~0.5s | **60x faster** |
| Install PyTorch | ~120s | ~8s | **15x faster** |

*Note: This project uses UV exclusively. pip is not required or supported.*

## UV Configuration

UV uses standard Python configuration:
- `.python-version` - specifies Python version (3.10)
- `requirements.txt` - package dependencies
- `pyproject.toml` - project metadata

## Troubleshooting

### UV not found after installation
```bash
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or restart shell
exec $SHELL
```

### Virtual environment not activating
```bash
# Make sure you're in project directory
cd /path/to/growingSparseSNN

# Activate explicitly
source .venv/bin/activate
```

### ROCm PyTorch installation with UV
```bash
# Use index URL for ROCm wheels
uv pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

### Check UV cache
```bash
# View cache size
du -sh ~/.cache/uv

# Clear cache if needed
rm -rf ~/.cache/uv
```

## Further Reading

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV vs pip comparison](https://github.com/astral-sh/uv#highlights)
- [Astral Blog](https://astral.sh/blog)

---This project uses UV exclusively as its package manager. Traditional pip workflows are not supported. UV provides all necessary functionality with significantly better performance

**Note**: UV is compatible with existing pip workflows. You can use both tools interchangeably, but UV is significantly faster for installation and dependency resolution.
