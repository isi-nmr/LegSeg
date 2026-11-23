#!/usr/bin/env bash
# Installation script for the LegSeg GUI (Linux).
# Usage: ./install.sh            (CUDA 12.8)
#        CUDA=cpu ./install.sh   (CPU-only torch)
set -euo pipefail
cd "$(dirname "$0")"

CUDA="${CUDA:-cu128}"

# 1) venv with Python 3.14
python3.14 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip

# 2) torch (must be installed before nnU-Net)
$PY -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA"

# 3) nnU-Net and GUI dependencies
$PY -m pip install -r requirements.txt

# 4) trained model (from GitHub Release)
MODEL_URL="https://github.com/isi-nmr/LegSeg/releases/download/model-v1/Dataset015_SARXfinal.zip"
if [ ! -d nnUNet_results/Dataset015_SARXfinal ]; then
    mkdir -p nnUNet_results
    curl -L --fail -o model.zip "$MODEL_URL"
    unzip -q model.zip -d nnUNet_results
    rm model.zip
fi

echo "Done. Run the GUI with: .venv/bin/python legseg.py"
