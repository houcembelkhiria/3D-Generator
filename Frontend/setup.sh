#!/bin/bash
set -e

echo "=== 3D Generator Frontend Setup ==="

if ! command -v node &>/dev/null; then
    echo "Error: Node.js not found. Install Node 18+."
    exit 1
fi
echo "Using: node $(node --version)"

echo "Installing dependencies..."
npm install -q

echo ""
echo "=== Setup complete! ==="
echo "To start the dev server:"
echo "  npm run dev"
