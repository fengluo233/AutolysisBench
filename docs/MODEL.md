# Released liver model

| Item | Value |
| --- | --- |
| Research identifier | A2F_10x |
| Checkpoint | Final epoch 200 generator; identical to research `latest` |
| Generator | Conditional nine-block ResNet, 64 base channels, instance normalization |
| Parameters | 14.684 million (rounded) |
| Bridge steps / tau | 5 / 0.01 |
| Inference | RGB, bicubic resize 256 × 256, normalize [-1,1], no flip |
| Output | 256 × 256 RGB PNG |
| Training | Batch 4, Adam 0.0002, beta=(0.5,0.999), 100+100 epochs |
| Loss weights | Adversarial=1, patch contrastive=1, bridge=1 |
| License | MIT |

The preset is transcribed from saved research training options. The historical seed and full case-level training manifest are not supplied; this release does not claim exact from-scratch reproduction of the published weights. Seed 42 is a release default, not a statement about the original training seed.

Training resizes to 286 and crops to 256. The original dataset implementation switches the first training stream's resize to the crop size during learning-rate decay; the second stream retains its original schedule. This behavior is preserved.

Predictions are plausible generated morphology rather than observed paired references. A random ten-patch check found that blank/low-tissue inputs can generate tissue-like textures. No tissue filter or quality score is included. Software smoke tests do not establish diagnostic utility or scientific performance.
