#!/bin/bash
set -e

echo "=== 3D Generator Backend v2 Setup ==="

# Detect Python
PYTHON=""
for cmd in python3.11 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Error: Python not found. Install Python 3.11+."
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version))"

# macOS system deps
if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &>/dev/null; then
    echo "Installing system dependencies via Homebrew..."
    brew install libmagic libjpeg libpng ninja 2>/dev/null || true
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi
source venv/bin/activate

# Install all deps + project
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -e . -q

# Build C++ extensions
echo "Building C++ extensions..."
(cd hy3dgen/texgen/custom_rasterizer && python setup.py install -q 2>&1 | tail -1)
(cd hy3dgen/texgen/differentiable_renderer && python setup.py install -q 2>&1 | tail -1)

echo ""
echo "=== Setup complete! ==="
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8001"
