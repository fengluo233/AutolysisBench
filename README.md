<div align="center">

# AutolysisBench

### Through the Schrödinger Bridge:<br>Benchmarking Antemortem Image Restoration from Postmortem Autolysis<br>to Enhance Forensic Diagnostics

**MICCAI 2026**

Shuang Hao<sup>1,4</sup> · Jiacheng Yue<sup>2</sup> · Yaxuan Zhao<sup>3,4</sup> · Fan Wang<sup>1,4</sup><br>
Jianhua Ma<sup>1,4</sup> · Erwen Huang<sup>2,✉</sup> · Chunfeng Lian<sup>3,4,✉</sup>

<sup>1</sup> School of Life Science and Technology, Xi'an Jiaotong University<br>
<sup>2</sup> Faculty of Forensic Medicine, Sun Yat-Sen University<br>
<sup>3</sup> School of Mathematics and Statistics, Xi'an Jiaotong University<br>
<sup>4</sup> Research Center for Intelligent Medical Equipment and Devices (IMED), Xi'an Jiaotong University

[![Paper](https://img.shields.io/badge/MICCAI_2026-Paper-315C93?style=flat-square)](https://papers.miccai.org/miccai-2026/1057-Paper0259.html)
[![arXiv](https://img.shields.io/badge/arXiv-2608.21813-B31B1B?style=flat-square)](https://arxiv.org/abs/2608.21813)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-D9A441?style=flat-square)](https://huggingface.co/datasets/fengluo/AutolysisBench)
[![Weights](https://img.shields.io/badge/Model-Weights-7B61A8?style=flat-square)](https://github.com/fengluo233/AutolysisBench/releases/tag/v1.0.0)
[![Code license](https://img.shields.io/badge/Code-MIT-34826E?style=flat-square)](LICENSE)

[Overview](#overview) · [Dataset](#dataset) · [Installation](#installation) · [Inference](#inference) · [Training](#training) · [中文说明](docs/README.zh-CN.md)

</div>

---

## Overview

**Can diagnostically useful tissue morphology be recovered from postmortem autolysis?** Autolysis disrupts nuclei, cell boundaries, and tissue organization. We study restoration toward plausible pre-autolysis morphology under **unpaired supervision**, where pixel-aligned antemortem targets are unavailable.

<p align="center">
  <img src="assets/framework.png" width="100%" alt="Framework from the paper: adjacent tissue blocks form unpaired domains; a Schrödinger Bridge models restoration; slide-level diagnostic distributions evaluate utility.">
</p>

<p align="center"><em>The framework from our paper: dataset construction, Schrödinger Bridge restoration, and diagnosis-oriented evaluation.</em></p>

- **AutoPath data:** homologous, unpaired liver tissue from adjacent blocks, processed immediately or after exposure to induce autolysis.
- **Stochastic restoration:** a Schrödinger Bridge transports the autolyzed image distribution toward the non-autolyzed domain.
- **Diagnostic evaluation:** the study examines diagnostic distribution consistency beyond generic image-level metrics such as FID.

This repository releases the **liver training and patch inference implementation**, the **epoch-200 generator**, and access to the **archived liver 10× patches**. The whole-slide evaluation pipeline, comparison methods, and other-organ experiments are outside this code release. Restored images are model-generated representations, not observed tissue ground truth.

| Resource | Where to find it |
|---|---|
| Paper and supplementary material | [MICCAI open access](https://papers.miccai.org/miccai-2026/1057-Paper0259.html) · [arXiv](https://arxiv.org/abs/2608.21813) |
| Liver 10× patches | [Hugging Face](https://huggingface.co/datasets/fengluo/AutolysisBench) · [Data guide](docs/DATASET.md) |
| Pretrained generator | [Release v1.0.0](https://github.com/fengluo233/AutolysisBench/releases/tag/v1.0.0) |
| Implementation and validation | [Model details](docs/MODEL.md) · [Validation record](docs/VALIDATION.md) |

## Dataset

The downloadable archive preserves the original `A2F_10x` training and test manifests. It contains **10,926 original 1024 × 1024 RGB PNG patches**, approximately **16.85 GB**, packaged as WebDataset TAR shards. Original PNG bytes are retained and checked with SHA-256.

| Split | Domain | Patches |
|---|---|---:|
| `trainA` | Autolyzed liver | 4,962 |
| `trainB` | Non-autolyzed liver | 4,964 |
| `testA` | Autolyzed liver | 500 |
| `testB` | Non-autolyzed liver | 500 |

> **Split and cohort accounting.** This archived release contains 79 cases, with case overlap between training and test data. Its manifest totals differ from the paper's 69-case, 10,924-patch cohort summary. The original split is preserved and the distinction is documented in the [data guide](docs/DATASET.md). These splits do not establish case-independent evaluation.

1. Sign in on the [Hugging Face dataset page](https://huggingface.co/datasets/fengluo/AutolysisBench) and submit the access form. **Requests are approved automatically.**
2. Authenticate the downloading machine and reconstruct the original four folders:

```bash
python -m pip install -r requirements-data.txt
hf auth login
python download_data.py --repo-id fengluo/AutolysisBench --output data/liver10x
```

Use `--splits testA testB` to download only the test patches. The output directory must not already exist. The downloader verifies every archive and extracted PNG, and records the dataset revision. Public identifiers are pseudonymous and consistent across the four splits.

```text
data/liver10x/
├── trainA/    # autolyzed training patches
├── trainB/    # non-autolyzed training patches
├── testA/     # autolyzed test patches
└── testB/     # non-autolyzed test patches
```

The dataset uses **CC BY 4.0**; code and model weights use **MIT**. See [dataset metadata, streaming access and limitations](docs/DATASET.md).

## Installation

Validated runtime: **Linux · Python 3.8 · PyTorch 1.9.0 / CUDA 11.1 · NVIDIA GPU**. The training and inference entry points do not support CPU or macOS GPU execution. Dataset downloading is independent of the model runtime.

```bash
git clone https://github.com/fengluo233/AutolysisBench.git
cd AutolysisBench
conda create -n autolysisbench python=3.8 -y
conda activate autolysisbench
python -m pip install -r requirements.txt
```

## Inference

### 1. Get the pretrained generator

```bash
mkdir -p checkpoints
curl -fL https://github.com/fengluo233/AutolysisBench/releases/download/v1.0.0/A2F_10x_epoch200_net_G.pth -o checkpoints/A2F_10x_epoch200_net_G.pth
curl -fL https://github.com/fengluo233/AutolysisBench/releases/download/v1.0.0/SHA256SUMS -o checkpoints/SHA256SUMS
(cd checkpoints && sha256sum -c SHA256SUMS)
```

The released file is byte-identical to the final epoch-200 generator of the research run. Its SHA-256 is:

```text
812e4ad73b19bdc5e4b56c5351cc85bce7497834d621fba29398a4a864f38983
```

### 2. Restore autolyzed patches

```bash
python infer.py \
  --input data/liver10x/testA \
  --checkpoint checkpoints/A2F_10x_epoch200_net_G.pth \
  --output results/liver --device cuda:0 --seed 42
```

Only input images and the generator are required. Inference converts each image to RGB, bicubic-resizes it to **256 × 256**, normalizes to `[-1, 1]`, and performs **five restoration steps**. The default output is `results/liver/fake_5/`; add `--save_steps` to save steps 1–4 as well.

<details>
<summary><b>Input, output and reproducibility details</b></summary>

- PNG, JPEG, TIFF, BMP and PPM images, including nested directories, are supported.
- `case/patch.jpg` becomes `fake_5/case/patch.jpg.png` to avoid extension collisions.
- The output directory must be new or empty. `run.json` records input paths, seed, checkpoint checksum and settings.
- Seeded reproduction requires the same input order, software and hardware; cross-platform numerical identity is not guaranteed.
- Use tissue-containing patches at the intended 10× extraction scale. There is no automatic blank-patch filter, and blank inputs can generate tissue-like textures.
- The output has 256 × 256 resolution and does not recover native 1024-pixel detail.

</details>

## Training

Use `trainA/` for autolyzed patches and `trainB/` for non-autolyzed patches. The collections are **unpaired**; matched filenames or a shared case identifier do not establish pixel registration.

```bash
python train.py --dataroot data/liver10x \
  --config configs/liver_a2f.json --name liver_training \
  --checkpoints_dir checkpoints --gpu_ids 0 --seed 42
```

| Setting | Released preset |
|---|---|
| Input preprocessing | Resize to 286 × 286, random crop to 256 × 256 |
| Batch size | 4 |
| Learning rate | 0.0002 |
| Schedule | 100 constant-rate epochs + 100 decay epochs |
| Bridge steps | 5 |

Explicit command-line options override `configs/liver_a2f.json`. Options, losses and network weights are saved under the experiment directory. Add `--batch_size 1 --num_threads 0 --max_steps 2` for a short functional check.

For new generalization experiments, construct case-disjoint partitions using `case_id` before training. The archived release retains the historical patch split described above.

`--continue_train` restores network weights only; optimizer and RNG states are not restored. The released generator alone supports inference, but does not provide a complete training checkpoint. For a newly trained model, use `checkpoints/liver_training/latest_net_G.pth` with `infer.py`.

## Validation

```bash
python tests/check_release.py \
  --checkpoint checkpoints/A2F_10x_epoch200_net_G.pth \
  --work-dir /tmp/autolysisbench-check
```

The work directory must not exist. The check exercises seeded five-step inference, two training updates, network save/load and invalid inputs using synthetic images. See the [validation record](docs/VALIDATION.md). The dataset release separately checks full PNG decoding, archive contents, hashes and exact split membership.

## Citation

If you use this work, please cite our [MICCAI 2026 paper](https://papers.miccai.org/miccai-2026/1057-Paper0259.html). [Download BibTeX](CITATION.bib).

```bibtex
@inproceedings{autolysis_schrodinger_bridge_2026,
  title     = {Through the Schrödinger Bridge: Benchmarking Antemortem Image Restoration from Postmortem Autolysis to Enhance Forensic Diagnostics},
  author    = {Hao, Shuang and Yue, Jiacheng and Zhao, Yaxuan and Wang, Fan and Ma, Jianhua and Huang, Erwen and Lian, Chunfeng},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026},
  note      = {Early Accept}
}
```

## Acknowledgments and license

Our code is **inspired by UNSB**, *Unpaired Image-to-Image Translation via Neural Schrödinger Bridge*, and adapted from its [public implementation](https://github.com/cyclomon/UNSB). Original copyright notices are retained. See [code provenance](docs/PROVENANCE.md) and the [upstream MIT license](licenses/upstream-MIT.txt).

Code and released model weights use [MIT](LICENSE). The liver patch dataset is separately released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The figure above is reproduced from our paper and retains its publication copyright.
