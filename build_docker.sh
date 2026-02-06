#!/bin/bash
# Build and run Docker container with ROCm support

echo "🐳 Building GrowingSparseSNN Docker image with ROCm..."

# Build image
docker build -t growing-snn:rocm -f docker/Dockerfile.rocm .

echo ""
echo "✅ Build complete!"
echo ""
echo "To run the container:"
echo "  docker run --rm -it \\"
echo "    --device=/dev/kfd --device=/dev/dri \\"
echo "    --group-add video --ipc=host \\"
echo "    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \\"
echo "    -v \$(pwd):/workspace \\"
echo "    growing-snn:rocm"
echo ""
echo "Or use docker-compose:"
echo "  cd docker && docker-compose up"
