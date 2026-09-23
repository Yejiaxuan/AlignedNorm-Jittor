# Installation

Run from the repository root:

```bash
conda create -n alignednorm-jittor python=3.9 -y
conda activate alignednorm-jittor
pip install -r requirements.txt
```

The port targets Python 3.9 and Jittor 1.3.11.0. Jittor compiles operators on first use, so the initial run may take longer. For GPU execution, use a compatible NVIDIA driver and CUDA environment.

The required Dassl-compatible runtime is included in `dassl/`; no separate `Dassl.pytorch` installation is needed. PyTorch is only required when converting original CLIP weights.

See the [README](../README.md#jittor-port) for weight conversion and training commands, and [DATASETS.md](DATASETS.md) for dataset layout.
