#!/bin/bash
# Sync dependencies using UV
# Updates requirements.txt from current venv packages

set -e

if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Install it with:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Make sure venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠ Virtual environment not activated"
    echo "Activating .venv..."
    source .venv/bin/activate
fi

echo "📦 Syncing dependencies with UV..."

# Freeze current environment to requirements.txt
uv pip freeze > requirements.txt

echo "✓ requirements.txt updated!"
echo ""
echo "Current packages:"
uv pip list
