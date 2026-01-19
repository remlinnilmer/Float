Floating Convolutional Blocks (FLOAT)

FLOAT is a stateless convolutional residual block designed to be a practical alternative to BatchNorm-style normalization. It uses Relative Energy Norm (REN) and a lightweight gate that applies bounded, groupwise gain/bias modulation from compact “wave descriptors” computed on the current activation field.

The goal is simple: stable training and reliable inference without running-statistics coupling, while remaining drop-in friendly for CNN backbones.

Why FLOAT

No running statistics: REN is per-sample and per-group, avoiding train/eval distribution mismatch from stored batch statistics.

Batch-size tolerant: designed to behave sensibly even at small batch sizes (including batch size 1).

Bounded adaptation: gain/bias are constrained and initialized near identity for stability.

Compact conditioning: descriptors are small, interpretable signals (moments + optional texture/roughness terms).

Deployment option: supports freeze-to-static mode for deterministic, cheaper inference.

What’s in this repo

FloatingBlock.py — FloatingConvBlockV1 + REN + gated modulation + config (standalone module)

cifar10Test.py — CIFAR-10 dual benchmark (FLOAT vs ResNet-18)

CIFAIR10Results.txt — example run log (configs, per-epoch metrics, best accuracy)

Example run:

python cifar10Test.py --epochs 300 --batch-size 512 --workers 8 --amp --amp-dtype bf16 --ema 0.9999 --mixup 0.2 --cutmix 1.0 --label-smoothing 0.1 --randaugment-n 2 --randaugment-m 9


Evaluation note:

The benchmark script uses BN-safe evaluation behavior: if a model contains BatchNorm2d, validation defaults to the RAW model (not EMA) to avoid BN running-stat artifacts.

Results:

See CIFAIR10Results.txt for a full example log from an A100 run.

