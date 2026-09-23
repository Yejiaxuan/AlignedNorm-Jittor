# Jittor Results

## Base-to-New

Entries are **Base / New / HM (%)**, averaged over three seeds; HM denotes the harmonic mean. Training uses 16 images per base class. MMRL and MMRL++ use decoupled evaluation; AlignedNorm uses unified inference.

| Dataset | Method | Training data | Test images | Paper | PyTorch AMP | PyTorch FP32 | Jittor FP32 (native) | Jittor FP32 (controlled) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EuroSAT | MMRL | 5 base classes, 16 images/class | Base 4,200 + New 3,900 | 96.10 / 72.33 / 82.54 | 96.10 / 72.33 / 82.54 | 96.47 / 74.53 / 84.09 | 96.60 / 74.43 / 84.08 | 96.80 / 71.27 / 82.09 |
| EuroSAT | MMRL++ | 5 base classes, 16 images/class | Base 4,200 + New 3,900 | 95.73 / 84.17 / 89.58 | 95.73 / 84.17 / 89.58 | 95.77 / 85.20 / 90.17 | 95.80 / 85.40 / 90.30 | 95.70 / 85.13 / 90.11 |
| EuroSAT | AlignedNorm | 5 base classes, 16 images/class | Base 4,200 + New 3,900 | 96.10 / 86.63 / 91.12 | 96.10 / 86.63 / 91.12 | 95.73 / 86.50 / 90.88 | 95.17 / 80.90 / 87.46 | 95.70 / 86.50 / 90.87 |
| DTD | MMRL | 24 base classes, 16 images/class | Base 864 + New 828 | 85.87 / 64.10 / 73.40 | 85.87 / 64.10 / 73.40 | 85.80 / 64.53 / 73.66 | 84.47 / 63.57 / 72.53 | 85.80 / 64.53 / 73.66 |
| DTD | MMRL++ | 24 base classes, 16 images/class | Base 864 + New 828 | 85.07 / 65.83 / 74.22 | 85.07 / 65.83 / 74.22 | 84.93 / 65.63 / 74.04 | 84.23 / 63.83 / 72.62 | 84.97 / 65.63 / 74.06 |
| DTD | AlignedNorm | 24 base classes, 16 images/class | Base 864 + New 828 | 84.63 / 65.23 / 73.67 | 84.63 / 65.23 / 73.67 | 84.53 / 64.40 / 73.11 | 83.90 / 62.43 / 71.59 | 84.53 / 64.33 / 73.06 |
| UCF101 | MMRL | 51 base classes, 16 images/class | Base 1,934 + New 1,849 | 88.30 / 79.63 / 83.74 | 88.30 / 79.63 / 83.74 | 88.43 / 79.87 / 83.93 | 87.90 / 78.93 / 83.17 | 88.40 / 79.87 / 83.92 |
| UCF101 | MMRL++ | 51 base classes, 16 images/class | Base 1,934 + New 1,849 | 87.37 / 80.43 / 83.76 | 87.37 / 80.43 / 83.76 | 87.77 / 79.77 / 83.58 | 87.23 / 79.03 / 82.93 | 87.73 / 79.77 / 83.56 |
| UCF101 | AlignedNorm | 51 base classes, 16 images/class | Base 1,934 + New 1,849 | 87.43 / 81.00 / 84.09 | 87.43 / 81.00 / 84.09 | 87.37 / 80.33 / 83.70 | 86.93 / 79.63 / 83.12 | 87.27 / 80.50 / 83.74 |
| OxfordPets | MMRL | 19 base classes, 16 images/class | Base 1,881 + New 1,788 | 95.97 / 97.50 / 96.73 | 95.97 / 97.50 / 96.73 | 95.87 / 97.47 / 96.66 | 95.63 / 97.43 / 96.52 | 95.83 / 97.47 / 96.64 |
| OxfordPets | MMRL++ | 19 base classes, 16 images/class | Base 1,881 + New 1,788 | 95.43 / 96.97 / 96.19 | 95.43 / 96.97 / 96.19 | 95.43 / 96.83 / 96.13 | 95.03 / 97.00 / 96.01 | 95.27 / 96.77 / 96.01 |
| OxfordPets | AlignedNorm | 19 base classes, 16 images/class | Base 1,881 + New 1,788 | 95.63 / 97.43 / 96.52 | 95.63 / 97.43 / 96.52 | 95.73 / 97.37 / 96.54 | 95.53 / 96.80 / 96.16 | 95.70 / 97.37 / 96.53 |
| Caltech101 | MMRL | 50 base classes, 16 images/class | Base 1,549 + New 916 | 98.83 / 94.33 / 96.53 | 98.83 / 94.33 / 96.53 | 98.83 / 94.33 / 96.53 | 98.97 / 94.73 / 96.80 | 98.83 / 94.30 / 96.51 |
| Caltech101 | MMRL++ | 50 base classes, 16 images/class | Base 1,549 + New 916 | 98.90 / 94.40 / 96.60 | 98.90 / 94.40 / 96.60 | 98.77 / 94.47 / 96.57 | 98.67 / 94.23 / 96.40 | 98.93 / 94.50 / 96.67 |
| Caltech101 | AlignedNorm | 50 base classes, 16 images/class | Base 1,549 + New 916 | 98.90 / 94.77 / 96.79 | 98.77 / 94.80 / 96.74 | 98.77 / 94.93 / 96.81 | 98.83 / 95.00 / 96.88 | 98.97 / 94.87 / 96.87 |
| OxfordFlowers | MMRL | 51 base classes, 16 images/class | Base 1,053 + New 1,410 | 98.97 / 76.97 / 86.59 | 98.97 / 76.97 / 86.59 | 98.83 / 76.70 / 86.37 | 98.80 / 76.83 / 86.44 | 98.83 / 76.63 / 86.33 |
| OxfordFlowers | MMRL++ | 51 base classes, 16 images/class | Base 1,053 + New 1,410 | 98.50 / 77.47 / 86.73 | 98.50 / 77.47 / 86.73 | 98.07 / 77.23 / 86.41 | 98.03 / 76.40 / 85.88 | 98.23 / 77.37 / 86.56 |
| OxfordFlowers | AlignedNorm | 51 base classes, 16 images/class | Base 1,053 + New 1,410 | 98.40 / 76.03 / 85.78 | 98.40 / 76.03 / 85.78 | 97.93 / 76.10 / 85.65 | 97.87 / 75.60 / 85.30 | 97.77 / 76.27 / 85.69 |
| FGVCAircraft | MMRL | 50 base classes, 16 images/class | Base 1,666 + New 1,667 | 46.13 / 37.47 / 41.35 | 46.13 / 37.47 / 41.35 | 45.97 / 37.10 / 41.06 | 45.37 / 37.20 / 40.88 | 45.97 / 37.13 / 41.08 |
| FGVCAircraft | MMRL++ | 50 base classes, 16 images/class | Base 1,666 + New 1,667 | 46.47 / 38.50 / 42.11 | 46.47 / 38.50 / 42.11 | 46.27 / 38.50 / 42.03 | 47.00 / 38.10 / 42.08 | 46.27 / 38.50 / 42.03 |
| FGVCAircraft | AlignedNorm | 50 base classes, 16 images/class | Base 1,666 + New 1,667 | 46.20 / 38.60 / 42.06 | 46.20 / 38.60 / 42.06 | 46.40 / 37.97 / 41.76 | 45.63 / 37.80 / 41.35 | 46.33 / 38.03 / 41.78 |
| StanfordCars | MMRL | 98 base classes, 16 images/class | Base 4,002 + New 4,039 | 81.30 / 74.83 / 77.93 | 81.30 / 74.83 / 77.93 | 81.20 / 74.83 / 77.89 | 81.47 / 74.63 / 77.90 | 81.23 / 74.80 / 77.88 |
| StanfordCars | MMRL++ | 98 base classes, 16 images/class | Base 4,002 + New 4,039 | 81.23 / 75.23 / 78.11 | 81.23 / 75.23 / 78.12 | 81.07 / 75.27 / 78.06 | 80.73 / 74.70 / 77.60 | 81.17 / 75.23 / 78.09 |
| StanfordCars | AlignedNorm | 98 base classes, 16 images/class | Base 4,002 + New 4,039 | 81.70 / 73.83 / 77.57 | 81.57 / 73.97 / 77.58 | 81.23 / 73.83 / 77.36 | 81.03 / 73.47 / 77.06 | 81.13 / 74.10 / 77.46 |
| Food101 | MMRL | 51 base classes, 16 images/class | Base 15,300 + New 15,000 | 90.57 / 91.53 / 91.05 | 90.57 / 91.53 / 91.05 | 90.47 / 91.53 / 91.00 | 90.37 / 91.57 / 90.96 | 90.47 / 91.53 / 91.00 |
| Food101 | MMRL++ | 51 base classes, 16 images/class | Base 15,300 + New 15,000 | 90.50 / 91.70 / 91.10 | 90.50 / 91.70 / 91.10 | 90.37 / 91.77 / 91.06 | 90.40 / 91.67 / 91.03 | 90.37 / 91.73 / 91.04 |
| Food101 | AlignedNorm | 51 base classes, 16 images/class | Base 15,300 + New 15,000 | 90.57 / 91.60 / 91.08 | 90.57 / 91.60 / 91.08 | 90.70 / 91.60 / 91.15 | 90.63 / 91.60 / 91.11 | 90.70 / 91.50 / 91.10 |
| SUN397 | MMRL | 199 base classes, 16 images/class | Base 9,950 + New 9,900 | 83.07 / 79.23 / 81.10 | 83.07 / 79.23 / 81.10 | 83.13 / 79.20 / 81.12 | 82.87 / 79.07 / 80.92 | 83.13 / 79.23 / 81.14 |
| SUN397 | MMRL++ | 199 base classes, 16 images/class | Base 9,950 + New 9,900 | 82.93 / 79.57 / 81.22 | 82.93 / 79.57 / 81.22 | 82.90 / 79.43 / 81.13 | 82.83 / 79.37 / 81.06 | 82.90 / 79.33 / 81.08 |
| SUN397 | AlignedNorm | 199 base classes, 16 images/class | Base 9,950 + New 9,900 | 82.93 / 79.13 / 80.99 | 82.93 / 79.13 / 80.99 | 83.07 / 79.10 / 81.03 | 83.07 / 78.93 / 80.95 | 82.97 / 79.13 / 81.00 |
| ImageNet | MMRL | 500 base classes, 16 images/class | Base 25,000 + New 25,000 | 77.70 / 71.20 / 74.31 | 77.70 / 71.20 / 74.31 | 77.70 / 71.17 / 74.29 | 77.97 / 71.33 / 74.50 | 77.40 / 71.23 / 74.19 |
| ImageNet | MMRL++ | 500 base classes, 16 images/class | Base 25,000 + New 25,000 | 77.60 / 71.40 / 74.37 | 77.60 / 71.40 / 74.37 | 77.63 / 71.37 / 74.37 | 77.70 / 71.43 / 74.44 | 77.83 / 71.53 / 74.55 |
| ImageNet | AlignedNorm | 500 base classes, 16 images/class | Base 25,000 + New 25,000 | 77.60 / 71.47 / 74.41 | 77.60 / 71.47 / 74.41 | 77.60 / 71.43 / 74.39 | 77.63 / 71.30 / 74.33 | 77.73 / 70.47 / 73.92 |

PyTorch FP32 and Jittor FP32 use the same configurations.

## Runtime and memory

NVIDIA GeForce RTX 3090, FP32, batch size 32, seed 1, 10 warm-up iterations and 30 measured iterations. Times are **ms/batch**; GPU memory is **MiB**. Each pair is **training / inference**.

| Dataset | Method | PyTorch precision | PyTorch time | PyTorch memory | Jittor precision | Jittor time | Jittor memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EuroSAT | MMRL | FP32 | 252.9 / 85.1 | 3,900 / 1,942 | FP32 | 254.0 / 90.0 | 5,448 / 3,538 |
| EuroSAT | MMRL++ | FP32 | 254.3 / 85.4 | 3,678 / 1,862 | FP32 | 257.2 / 90.0 | 5,124 / 3,192 |
| EuroSAT | AlignedNorm | FP32 | 256.5 / 85.4 | 4,208 / 2,024 | FP32 | 261.1 / 90.7 | 5,474 / 3,328 |
| DTD | MMRL | FP32 | 264.8 / 85.5 | 4,012 / 1,954 | FP32 | 268.9 / 89.6 | 5,522 / 3,210 |
| DTD | MMRL++ | FP32 | 267.7 / 85.9 | 3,988 / 1,900 | FP32 | 274.0 / 89.9 | 5,428 / 3,194 |
| DTD | AlignedNorm | FP32 | 270.3 / 85.9 | 4,796 / 2,254 | FP32 | 277.9 / 91.0 | 5,894 / 3,330 |
| UCF101 | MMRL | FP32 | 291.6 / 86.1 | 4,472 / 2,092 | FP32 | 293.3 / 89.9 | 5,858 / 3,190 |
| UCF101 | MMRL++ | FP32 | 292.8 / 86.0 | 4,452 / 2,088 | FP32 | 296.5 / 90.2 | 5,816 / 3,174 |
| UCF101 | AlignedNorm | FP32 | 295.7 / 86.1 | 5,256 / 2,290 | FP32 | 300.3 / 91.1 | 6,282 / 3,234 |
| OxfordPets | MMRL | FP32 | 261.8 / 84.6 | 3,940 / 1,938 | FP32 | 262.4 / 89.7 | 5,374 / 3,216 |
| OxfordPets | MMRL++ | FP32 | 263.3 / 84.8 | 3,918 / 1,898 | FP32 | 267.4 / 90.0 | 5,358 / 3,200 |
| OxfordPets | AlignedNorm | FP32 | 263.5 / 84.9 | 4,702 / 2,258 | FP32 | 271.8 / 91.0 | 5,708 / 3,336 |
| Caltech101 | MMRL | FP32 | 316.5 / 85.0 | 4,472 / 2,092 | FP32 | 298.8 / 90.4 | 5,904 / 3,196 |
| Caltech101 | MMRL++ | FP32 | 330.5 / 85.0 | 4,452 / 2,088 | FP32 | 302.2 / 90.6 | 5,812 / 3,180 |
| Caltech101 | AlignedNorm | FP32 | 324.5 / 85.1 | 5,256 / 2,290 | FP32 | 303.8 / 91.7 | 6,278 / 3,240 |
| OxfordFlowers | MMRL | FP32 | 287.9 / 84.8 | 4,472 / 2,092 | FP32 | 289.3 / 88.5 | 5,862 / 3,196 |
| OxfordFlowers | MMRL++ | FP32 | 289.3 / 84.8 | 4,452 / 2,088 | FP32 | 294.1 / 89.6 | 5,820 / 3,180 |
| OxfordFlowers | AlignedNorm | FP32 | 293.5 / 84.8 | 5,256 / 2,290 | FP32 | 299.3 / 90.8 | 6,286 / 3,240 |
| FGVCAircraft | MMRL | FP32 | 287.3 / 84.9 | 4,472 / 2,092 | FP32 | 287.6 / 89.8 | 5,904 / 3,196 |
| FGVCAircraft | MMRL++ | FP32 | 290.1 / 84.8 | 4,452 / 2,088 | FP32 | 291.4 / 89.9 | 5,812 / 3,180 |
| FGVCAircraft | AlignedNorm | FP32 | 295.8 / 84.9 | 5,256 / 2,290 | FP32 | 297.4 / 90.6 | 6,278 / 3,240 |
| StanfordCars | MMRL | FP32 | 347.1 / 84.9 | 5,312 / 2,218 | FP32 | 330.1 / 89.9 | 6,674 / 3,168 |
| StanfordCars | MMRL++ | FP32 | 349.8 / 85.1 | 5,274 / 2,194 | FP32 | 333.6 / 89.9 | 6,554 / 3,152 |
| StanfordCars | AlignedNorm | FP32 | 357.0 / 85.1 | 6,056 / 2,476 | FP32 | 337.6 / 90.6 | 6,962 / 3,288 |
| Food101 | MMRL | FP32 | 293.4 / 85.6 | 4,472 / 2,092 | FP32 | 290.0 / 89.3 | 5,862 / 3,196 |
| Food101 | MMRL++ | FP32 | 289.8 / 85.0 | 4,452 / 2,088 | FP32 | 294.1 / 89.4 | 5,820 / 3,180 |
| Food101 | AlignedNorm | FP32 | 295.7 / 85.5 | 5,256 / 2,290 | FP32 | 297.8 / 90.4 | 6,286 / 3,240 |
| SUN397 | MMRL | FP32 | 455.8 / 85.1 | 7,030 / 2,234 | FP32 | 406.7 / 89.1 | 8,078 / 3,538 |
| SUN397 | MMRL++ | FP32 | 465.5 / 85.2 | 7,012 / 2,230 | FP32 | 411.4 / 89.8 | 7,926 / 3,522 |
| SUN397 | AlignedNorm | FP32 | 467.6 / 84.9 | 7,772 / 2,640 | FP32 | 416.4 / 90.4 | 8,186 / 3,522 |
| ImageNet | MMRL | FP32 | 786.7 / 85.3 | 12,208 / 3,734 | FP32 | 623.7 / 89.6 | 12,032 / 4,688 |
| ImageNet | MMRL++ | FP32 | 786.6 / 84.8 | 12,190 / 3,730 | FP32 | 630.2 / 90.4 | 12,030 / 4,672 |
| ImageNet | AlignedNorm | FP32 | 793.4 / 84.8 | 13,212 / 4,670 | FP32 | 632.7 / 91.2 | 12,532 / 4,672 |

Jittor inference is slightly slower and uses more GPU memory. Training speed is comparable to PyTorch and is faster on several larger datasets.
