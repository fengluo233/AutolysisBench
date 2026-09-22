# Code provenance

AutolysisBench is **inspired by UNSB** and adapts the implementation of *Unpaired Image-to-Image Translation via Neural Schrödinger Bridge* (ICLR 2024), by Beomsu Kim, Gihyun Kwon, Kwanyoung Kim and Jong Chul Ye.

- [Original code](https://github.com/cyclomon/UNSB)
- [Original paper](https://arxiv.org/abs/2305.15086)
- Research checkout base revision: `d1f644f7777e19d5afe5aea3e5cb4bd3afd9b88b`.

This release was extracted from the working research checkout, including local changes to its conditional-network and bridge-model modules. It is not an unchanged upstream checkout.

Network, loss and data-transform modules retain the research implementation. Packaging changes include input-only inference, training with console/JSON logging, a liver preset, explicit seeds/devices, validation and integration tests. Inference supplies source images in unused reference slots and retains the original initialization and forward passes to preserve stochastic execution. No architecture or loss redesign is claimed in this release.

Original source comments acknowledging CycleGAN, CUT and StyleGAN-derived implementations are retained. The upstream MIT notice is preserved verbatim in `licenses/upstream-MIT.txt`; the root license covers released code and weights.

The release disables cuDNN benchmarking and enables deterministic cuDNN selection for seeded runs. Reference comparisons use the same backend settings. These settings improve repeatability within the tested environment; they do not guarantee identical results across different hardware or software versions.
