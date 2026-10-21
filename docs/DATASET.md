# Liver 10× patch dataset

The public dataset is hosted at **[fengluo/AutolysisBench](https://huggingface.co/datasets/fengluo/AutolysisBench)**. This release contains the original liver 10× training and test patches associated with the released `A2F_10x` model, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code and model weights retain their existing MIT license.

| Split | Domain | Patches | Cases |
|---|---|---:|---:|
| `trainA` | Autolyzed | 4,962 | 79 |
| `trainB` | Non-autolyzed | 4,964 | 79 |
| `testA` | Autolyzed | 500 | 78 |
| `testB` | Non-autolyzed | 500 | 79 |

Total: **10,926 patches from 79 cases**. The release uses pseudonymous `case_id` and sample filenames consistently across all splits. Original accession-style filenames and internal storage paths are not included.

## Request access

Sign in on the Hugging Face dataset page and submit its access form. Requests are **approved automatically**. On the downloading machine, run `hf auth login` with a token from that account before downloading.

## Download the original PNGs

```bash
python -m pip install -r requirements-data.txt
python download_data.py --repo-id fengluo/AutolysisBench --output data/liver10x
```

Run these commands from the [code repository](https://github.com/fengluo233/AutolysisBench). The downloader verifies each archive and PNG with SHA-256, preserves the original PNG bytes, and reconstructs `trainA/`, `trainB/`, `testA/`, and `testB/`. The output directory must not already exist. Use `--splits testA testB` to download only the test data. `--revision` accepts a dataset tag or commit; each download resolves one revision and records it in `download_record.json`.

Train with the extracted folders:

```bash
python train.py --dataroot data/liver10x \
  --config configs/liver_a2f.json --name liver_training \
  --checkpoints_dir checkpoints --gpu_ids 0 --seed 42
```

## Read directly from Hugging Face

The data are uncompressed TAR shards in WebDataset format; each sample has a `.png` and matching `.json` record. No extra JPEG compression or resizing is applied.

```python
from datasets import load_dataset

train_a = load_dataset("fengluo/AutolysisBench", split="trainA", streaming=True)
sample = next(iter(train_a))
image = sample["png"]
metadata = sample["json"]
```

Install the optional `datasets` library to use this example. The four split names are `trainA`, `trainB`, `testA`, and `testB`. Each JSON record includes `sample_id`, `case_id`, `domain`, `split`, `width`, `height`, and the PNG's `sha256`.

## Integrity and split limitations

All 10,926 PNGs were fully decoded and checked as 1024 × 1024 RGB images. The original PNG payloads total 16,848,298,432 bytes (16.85 GB). Their bytes are preserved in the archives. SHA-256 checking found 0 duplicate-content groups, including 0 groups spanning training and test data.

The original patch-level membership is preserved. There are 79 distinct cases; 79 occur in both the combined training and combined test splits. The splits have no overlapping source file paths, but they are not case-independent. These historical splits must not be described as a held-out-case benchmark. Use `case_id` to construct case-disjoint partitions for new generalization experiments. Autolyzed and non-autolyzed patches are unpaired: a shared case ID does not establish pixel-level or patch-level registration.

The release counts are derived from the preserved manifests, rather than reconstructed from summary tables in a poster. In particular, `trainB` contains 4,964 patches; the published manifest is not reduced to 4,962 for numerical symmetry. The 79-case count describes this release and should not be substituted for a differently defined cohort in the paper without explanation.

`release_manifest.jsonl` records each public sample and its archive, byte size, and checksum. `release_summary.json` records counts, overlap, and archive checksums. `SHA256SUMS` covers the published files. Generated restoration images, whole-slide files, other organs, and diagnostic annotations are outside this patch release.

## Image scale

The original patches are 1024 × 1024 pixels at the study's 10× extraction scale. The training preset resizes to 286 × 286 and randomly crops 256 × 256. Inference resizes to 256 × 256 using bicubic interpolation. The 256-pixel model output does not preserve native 1024-pixel detail.

## Citation

Please cite **Through the Schrödinger Bridge: Benchmarking Antemortem Image Restoration from Postmortem Autolysis to Enhance Forensic Diagnostics** (Hao et al., MICCAI 2026). See [CITATION.bib](../CITATION.bib) for the full author list.
