#!/usr/bin/env bash
# One-time environment setup inside WSL2 (Ubuntu).
# Run from the repo root:  bash scripts/setup_wsl.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/grademl}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Python 3.12: coremltools only ships native wheels (needed to save
# .mlpackage weights) up to cp313, and the distro Python may be newer.
# uv fetches a standalone 3.12 without touching the system Python.
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --clear --python 3.12 "$VENV"
source "$VENV/bin/activate"

# torch 2.7.x: newest release coremltools 9.0 is tested against, and the
# first with Blackwell (sm_120) kernels in the cu128 build.
uv pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
uv pip install -r "$REPO/requirements.txt"

# Make src/ importable. `pip install -e .` fails on /mnt/* (DrvFs rejects
# setuptools' temp files), so register the path with a .pth file instead.
echo "$REPO/src" > "$(python -c 'import site; print(site.getsitepackages()[0])')/grademl.pth"

# Data lives on the WSL ext4 filesystem: reading thousands of small images
# through /mnt/f is several times slower.
mkdir -p "$HOME/grademl-data"
grep -q GRADEML_DATA "$HOME/.bashrc" || echo "export GRADEML_DATA=\$HOME/grademl-data" >> "$HOME/.bashrc"

python "$REPO/scripts/check_env.py"
echo
echo "Done. Activate later with: source $VENV/bin/activate"
