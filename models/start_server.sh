#!/bin/bash
# DocuNative - llama.cpp Server Setup Script
# Clones llama.cpp, compiles with Metal/CUDA/CPU fallback, starts server on port 8080
#
# Build documentation: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
#
set -e

MODEL_TYPE="${1:-global}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/llama.cpp"
SERVER_PORT=8080

# Map model type to filename
case "$MODEL_TYPE" in
    "global")
        MODEL_FILE="tiny-aya-global-q4_k_m.gguf"
        ;;
    "earth")
        MODEL_FILE="tiny-aya-earth-q4_k_m.gguf"
        ;;
    *)
        echo "Error: Invalid model type '$MODEL_TYPE'. Use 'global' or 'earth'."
        exit 1
        ;;
esac

MODEL_PATH="$SCRIPT_DIR/$MODEL_FILE"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Please run: python models/pull_models.py"
    exit 1
fi

echo "=========================================="
echo "DocuNative - Inference Server Setup"
echo "=========================================="
echo "Model: $MODEL_FILE"
echo "Port: $SERVER_PORT"
echo ""

# Clone llama.cpp if not already present
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/3] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$REPO_DIR"
    echo "✓ Cloned llama.cpp"
else
    echo "[1/3] Using existing llama.cpp directory"
fi

cd "$REPO_DIR"

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

# Compile with appropriate backend using CMake
echo "[2/3] Compiling llama.cpp with CMake..."

if [ "$OS" = "Darwin" ]; then
    # macOS - use Metal acceleration
    echo "  → Detected macOS, building with Metal support..."
    cmake -B build -DGGML_METAL=1
    cmake --build build -j$(sysctl -n hw.ncpu)
elif [ "$OS" = "Linux" ]; then
    # Linux - try CUDA, fallback to CPU
    if command -v nvcc &> /dev/null || [ -d "/usr/local/cuda" ]; then
        echo "  → Detected CUDA available, building with CUDA support..."
        cmake -B build -DGGML_CUDA=1
        cmake --build build -j$(nproc)
    else
        echo "  → No CUDA detected, building for CPU..."
        cmake -B build
        cmake --build build -j$(nproc)
    fi
else
    echo "  → Unsupported OS: $OS, building for CPU..."
    cmake -B build
    cmake --build build -j4
fi

echo "✓ Compilation complete"

# Verify port is free
if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Error: Port $SERVER_PORT already in use"
    exit 1
fi

# Start server
echo "[3/3] Starting llama-server..."
echo ""
echo "Server will be available at: http://localhost:$SERVER_PORT"
echo "Health check: http://localhost:$SERVER_PORT/health"
echo "Press Ctrl+C to stop the server"
echo ""

# Start server
./build/bin/llama-server \
    --model "$MODEL_PATH" \
    --port "$SERVER_PORT" \
    --host 0.0.0.0 \
    --ctx-size 4096 \
    --n-gpu-layers 99
