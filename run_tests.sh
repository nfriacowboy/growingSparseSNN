#!/bin/bash
# Run tests with coverage

echo "🧪 Running GrowingSparseSNN tests..."

# Run pytest with coverage
pytest tests/ -v \
    --cov=src \
    --cov-report=html \
    --cov-report=term \
    --color=yes

echo ""
echo "✅ Tests complete!"
echo "View coverage report: open htmlcov/index.html"
