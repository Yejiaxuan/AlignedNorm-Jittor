# AlignedNorm-Jittor

Jittor implementation of [AlignedNorm](https://github.com/QByteM/AlignedNorm). See the original repository for the paper, method and dataset preparation.

The port includes Jittor versions of CLIP, AlignedNorm, MMRL, MMRL++ and the required Dassl runtime. It currently supports FP32.

## Installation

```bash
conda create -n alignednorm-jittor python=3.9 -y
conda activate alignednorm-jittor
pip install -r requirements.txt
```

The tested environment uses Jittor 1.3.11.0. Jittor compiles operators during the first run.

Convert the original CLIP ViT-B/16 checkpoint once:

```bash
pip install torch
python tools/convert_clip_weights.py clip \
  --input /path/to/ViT-B-16.pt \
  --output /path/to/ViT-B-16.pkl
```

## Running

```bash
export DATA_ROOT="/path/to/your/datasets"
export JITTOR_CLIP_WEIGHTS="/path/to/ViT-B-16.pkl"

# One Base-to-New experiment
SEEDS=1 bash scripts/alignednorm/base2new_train.sh eurosat
SEEDS=1 bash scripts/alignednorm/base2new_test.sh eurosat
```

The original experiment entry points are also available:

```bash
bash base_to_novel.sh
bash cross_datasets.sh
bash few_shot.sh
```

Use the same `SEEDS` value for training and evaluation. See the [original running instructions](https://github.com/QByteM/AlignedNorm#running) for experiment details.

## Results

[Jittor results](docs/RESULTS.md) include Base-to-New accuracy and RTX 3090 runtime and memory measurements.

## Acknowledgement

This repository is based on [QByteM/AlignedNorm](https://github.com/QByteM/AlignedNorm) and retains its [MIT license](LICENSE).
