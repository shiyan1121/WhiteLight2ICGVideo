"""White-light to ICG video prediction package."""

from __future__ import annotations

import os

# Windows scientific Python stacks can load duplicate OpenMP runtimes when
# MATLAB/Anaconda/PyTorch libraries are on the same PATH. Set this before
# importing torch/numpy submodules so command-line scripts remain usable.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
