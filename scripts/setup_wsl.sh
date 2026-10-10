#!/usr/bin/env bash
# One-time environment setup inside WSL2 (Ubuntu).
# Run from the repo root:  bash scripts/setup_wsl.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/grademl}"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel

# Blackwell (sm_120) needs a CUDA 12.8+ build of PyTorch.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .

# Data lives on the WSL ext4 filesystem: reading thousands of small images
# through /mnt/f is several times slower.
mkdir -p "$HOME/grademl-data"
grep -q GRADEML_DATA "$HOME/.bashrc" || echo "export GRADEML_DATA=\$HOME/grademl-data" >> "$HOME/.bashrc"

python scripts/check_env.py
echo
echo "Done. Activate later with: source $VENV/bin/activate"
