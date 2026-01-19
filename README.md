Floating Convolutional Blocks (FLOAT)

FLOAT is a stateless CNN residual block that replaces BatchNorm-style running statistics with Relative Energy Norm (REN) and bounded, groupwise gain/bias modulation derived from compact “wave descriptors” of the current normalized activation field.

This repo includes the core block implementation and CIFAR-10 dual-benchmark script (FLOAT vs ResNet-18) with BN-safe evaluation.

Files

FloatingBlock.py — FloatingConvBlockV1 + REN + gated modulation + config (standalone module).

cifar10TestA100.py — CIFAR-10 dual-benchmark script tuned for A100/AMP and BN-safe eval defaults.

CIFAIR10Results.txt — Example run log (A100) showing configs, per-epoch metrics, and best accuracy.


Notes:

The script runs FLOAT first, then ResNet18 under the same recipe.

BN-safe evaluation: if the model contains BatchNorm2d, validation uses the RAW model by default (avoids EMA + BN running-stat mismatch under heavy Mixup/CutMix).

Results

See CIFAIR10Results.txt for a full example log from an A100 run (device + config printed at start; per-epoch metrics; best validation accuracy).

Using FLOAT in your own model

Import the block and config from FloatingBlock.py:

from FloatingBlock import FloatingConvBlockV1, FloatingBlockConfig


Instantiate blocks and configure REN + bounded modulation via FloatingBlockConfig (gate groups, gain/bias bounds, optional descriptors like TV/Sobel/entropy/kurtosis, etc.).
