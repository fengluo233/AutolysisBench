# AutolysisBench

This is the official repository for our MICCAI 2026 Early Accept paper:

**Through the Schrödinger Bridge: Benchmarking Antemortem Image Restoration from Postmortem Autolysis to Enhance Forensic Diagnostics**

Shuang Hao<sup>1,4</sup>, Jiacheng Yue<sup>2</sup>, Yaxuan Zhao<sup>3,4</sup>, Fan Wang<sup>1,4</sup>, Jianhua Ma<sup>1,4</sup>, Erwen Huang<sup>2,*</sup>, Chunfeng Lian<sup>3,4,*</sup>

<sup>1</sup> School of Life Science and Technology, Xi'an Jiaotong University<br>
<sup>2</sup> Faculty of Forensic Medicine, Sun Yat-Sen University<br>
<sup>3</sup> School of Mathematics and Statistics, Xi'an Jiaotong University<br>
<sup>4</sup> IMED, Xi'an Jiaotong University

English · [中文说明](docs/README.zh-CN.md) · [Model details](docs/MODEL.md)

## Release scope

This release provides **liver model training, patch-level inference, and pretrained generator weights**. Whole-slide processing, comparison methods, other-organ experiments and raw data are not included. Dataset access instructions remain pending.

## Abstract

Postmortem autolysis causes severe morphological degradation in forensic histopathology, making diagnosis more difficult and less reproducible. We formulate autolysis restoration as an unpaired image restoration task and benchmark restoration from autolyzed liver histopathology images toward plausible antemortem morphology. Our study introduces a physical tissue-splitting dataset construction protocol, evaluates Schrödinger Bridge based restoration against representative baselines, and shows that conventional image-level metrics such as FID can be poorly aligned with diagnostic utility.

## Installation

Validated runtime: Linux, Python 3.8, PyTorch 1.9.0/CUDA 11.1 and an NVIDIA GPU. These entry points do not support CPU or macOS GPU execution. See [validation details](docs/VALIDATION.md).

```bash
git clone https://github.com/fengluo233/AutolysisBench.git
cd AutolysisBench
conda create -n autolysisbench python=3.8 -y
conda activate autolysisbench
python -m pip install -r requirements.txt
```

## Pretrained weights

Download the generator and checksum from [v1.0.0](https://github.com/fengluo233/AutolysisBench/releases/tag/v1.0.0):

```bash
mkdir -p checkpoints
curl -fL https://github.com/fengluo233/AutolysisBench/releases/download/v1.0.0/A2F_10x_epoch200_net_G.pth -o checkpoints/A2F_10x_epoch200_net_G.pth
curl -fL https://github.com/fengluo233/AutolysisBench/releases/download/v1.0.0/SHA256SUMS -o checkpoints/SHA256SUMS
(cd checkpoints && sha256sum -c SHA256SUMS)
```

The file is byte-identical to the research run's final epoch-200 generator. SHA-256:

```text
812e4ad73b19bdc5e4b56c5351cc85bce7497834d621fba29398a4a864f38983
```

## Restore patches

Only autolyzed images and the generator weights are needed. No target-domain images, training dataset or visualization server is required.

```bash
python infer.py \
  --input /path/to/autolyzed_patches \
  --checkpoint checkpoints/A2F_10x_epoch200_net_G.pth \
  --output results/liver --device cuda:0 --seed 42
```

Nested directories and PNG/JPEG/TIFF/BMP/PPM files are supported. Each image is converted to RGB, resized to **256 × 256** with bicubic interpolation, normalized to `[-1, 1]`, and restored in five steps. Outputs are in `results/liver/fake_5/`; `case/patch.jpg` becomes `fake_5/case/patch.jpg.png` to avoid extension collisions. `run.json` records the inputs, seed, checksum and settings. Add `--save_steps` to also save steps 1–4.

The output directory must be new or empty. Repeatability requires the same sorted input sequence, seed, software and hardware; cross-platform numerical identity is not promised.

Use tissue-containing patches at the intended 10× scale. The research pipeline extracted 1024 × 1024 patches and resized them before inference; this release does not restore native 1024-resolution detail. There is no automatic blank-patch filter: **blank or low-tissue inputs can generate tissue-like textures**. Filter and inspect input patches before use.

## Train

Prepare unpaired collections with no filename-matching requirement:

```text
data_root/
├── trainA/   # autolyzed liver patches
└── trainB/   # non-autolyzed liver patches
```

Split training and evaluation data by case before patch extraction; the code does not create or enforce a study split.

```bash
python train.py --dataroot /path/to/data_root \
  --config configs/liver_a2f.json --name liver_training \
  --checkpoints_dir checkpoints --gpu_ids 0 --seed 42
```

The default config is `configs/liver_a2f.json`; explicit CLI flags override its values. The preset uses batch size 4, learning rate 0.0002, 100 constant-rate epochs followed by 100 decay epochs, and five bridge steps. Effective options, losses and network weights are saved under the experiment directory.

Add `--batch_size 1 --num_threads 0 --max_steps 2` for a short functional check. Existing output requires `--continue_train`, which restores network weights only, not optimizer or RNG state; this is not an exact resume. The released generator alone is sufficient for inference but not full training continuation.

For your trained model, pass `--checkpoint checkpoints/liver_training/latest_net_G.pth` to `infer.py`.

## Validation

```bash
python tests/check_release.py \
  --checkpoint checkpoints/A2F_10x_epoch200_net_G.pth \
  --work-dir /tmp/autolysisbench-check
```

The work directory must not exist. Tests generate synthetic images, check seeded five-step inference, train for two updates, reload all four networks and exercise invalid inputs. See [the validation record](docs/VALIDATION.md).

## Acknowledgments and license

Our code is **inspired by UNSB**, *Unpaired Image-to-Image Translation via Neural Schrödinger Bridge*. The model implementation is adapted from its [public code](https://github.com/cyclomon/UNSB). See [code provenance](docs/PROVENANCE.md).

Code and released model weights use the [MIT License](LICENSE). Original copyright notices are retained in [the upstream license](licenses/upstream-MIT.txt). This license does not grant access to, or rights over, the research data.

## Citation

Please cite the paper when using this work. [Download BibTeX](CITATION.bib).

```bibtex
@inproceedings{autolysis_schrodinger_bridge_2026,
  title     = {Through the Schrödinger Bridge: Benchmarking Antemortem Image Restoration from Postmortem Autolysis to Enhance Forensic Diagnostics},
  author    = {Hao, Shuang and Yue, Jiacheng and Zhao, Yaxuan and Wang, Fan and Ma, Jianhua and Huang, Erwen and Lian, Chunfeng},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026},
  note      = {Early Accept}
}
```
