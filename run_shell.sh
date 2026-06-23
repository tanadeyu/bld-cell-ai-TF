#!/bin/bash
# Blood Cell AI - インタラクティブシェル (ROCm + Docker)
IMAGE="tf-rocm-bloodcell"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

if ! docker images --format '{{.Repository}}' | grep -q "^$IMAGE$"; then
    echo "初回ビルド..."
    docker build -t $IMAGE .
fi

docker rm -f tf-shell 2>/dev/null

docker run --rm -it \
  --device=/dev/dxg \
  --memory="18g" --memory-swap="18g" \
  --name tf-shell \
  -v "$PROJECT_DIR":/workspace \
  -v /opt/rocm:/opt/rocm:ro \
  -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -w /workspace \
  $IMAGE \
  bash
