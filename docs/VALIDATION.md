# Release validation

Validated on 2026-09-22 using Linux, an NVIDIA RTX 3090, Python 3.8, PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111, NumPy 1.24.4, Pillow 10.4.0 and packaging 25.0.

Testing used a separate source checkout and temporary output directory. A virtual environment inherited the installed research runtime; `pip install --no-index -r requirements.txt` verified that the pinned direct dependencies were satisfied. A fresh internet installation on a separate machine was not tested.

| Check | Result |
| --- | --- |
| Published generator checksum | Matches the research latest and epoch-200 files |
| Seeded repeat inference | Identical PNGs for two synthetic inputs at all five steps |
| Reference implementation comparison | Pixel-identical at all five steps on the same two inputs, with matched seed and backend settings |
| Training smoke test | Two updates; all recorded losses finite |
| Generator parameter update | Maximum absolute change 0.00041078688809648156 |
| Checkpoint round trip | G, F, D and E all reload strictly with finite tensors |
| Newly trained generator | Successfully used by the public inference entry point |
| Error handling | Empty input, missing checkpoint, corrupt image and nonempty output rejected |
| Network/transform source inspection | Retained network, loss and transform modules match the research source structurally |

cuDNN deterministic selection is enabled and benchmarking disabled. The original benchmarking setting produced repeat-run differences of at most one 8-bit intensity level in this check; reference comparisons use the same fixed settings as the release. Cross-platform determinism is not asserted.

A separate qualitative check ran the final generator on ten randomly sampled research liver patches. It confirmed usable loading/output and exposed the documented blank-input behavior. Those images and case identifiers are not distributed. This check is not a held-out scientific evaluation.

No full 200-epoch retraining, independent dataset evaluation, macOS GPU support or CPU support was claimed or tested. The test script uses synthetic data and writes outside the repository; it does not require patient data.
