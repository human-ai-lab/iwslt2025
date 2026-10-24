# NAIST Simultaneous Speech Translation System for IWSLT 2025

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ACL%20Anthology-brightgreen)](https://aclanthology.org/2025.iwslt-1.39/)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-blue)](https://huggingface.co/tan-ht/naist-iwslt2025)

**English → German · Chinese · Japanese**

Whisper + Qwen · LoRA adaptation · Local-agreement decoding · SHAS segmentation

</div>

The NAIST submission to the **IWSLT 2025 Simultaneous Speech Translation Track**.
This repository provides training and inference code, evaluation configurations,
and links to the released models for all three language pairs.

**[Quick Start](#-quick-start)** · **[Models](#models)** ·
**[Data](#data-layout)** · **[Inference](#inference)** ·
**[Training](#model-training)** · **[Citation](#citation)**

---

## 🚀 Quick Start

### 1. Set up the environment

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/ha3ci-lab/iwslt2025.git
cd iwslt2025
uv sync --frozen
source .venv/bin/activate
```

The tested environment uses **Linux x86-64, Python 3.10, and an NVIDIA GPU**
with a driver compatible with the CUDA 12 dependencies in `uv.lock`.
The repository vendors the Fairseq and SimulEval versions used by the system;
keep these versions when reproducing results.

### 2. Download the models

Install the Hugging Face CLI in an isolated tool environment, then download the
three language models (approximately **13.3 GB** in total):

```bash
uv tool install huggingface_hub
hf download tan-ht/naist-iwslt2025 --local-dir checkpoints
```

For just English → German (approximately **4.43 GB**):

```bash
hf download tan-ht/naist-iwslt2025 \
  --include "en-de/*" \
  --local-dir checkpoints
```

Set the checkpoint paths to the downloaded **directories** so the loader reads
both the adapter and the compact checkpoint:

```bash
export CKPT_DE="$PWD/checkpoints/en-de"
export CKPT_ZH="$PWD/checkpoints/en-zh"
export CKPT_JA="$PWD/checkpoints/en-ja"
```

> The release contains adapters and custom model components. The Qwen and
> Whisper base models are downloaded separately on first use, so allow
> additional disk space and network access. See [Models](#models) for details.

### 3. Prepare evaluation data and run

Prepare the Fairseq configuration, vocabulary, and evaluation lists described
in [Data layout](#data-layout). Datasets and evaluation audio are obtained
separately.

Then run a short English → German evaluation on GPU `0`:

```bash
export DATA_ROOT="$PWD/data"
END_INDEX=10 bash script/simuleval_LA.sh en de 0
```

This evaluates the first 10 examples across the configured source chunk sizes.
Results are written under `exp_results/results_simul/de/`. Replace `de` with
`zh` or `ja` after downloading that model and preparing its evaluation data.

## Models

All three models are hosted in
[**tan-ht/naist-iwslt2025**](https://huggingface.co/tan-ht/naist-iwslt2025).

| Translation direction | Model files | Download size |
| --- | --- | ---: |
| English → German | [en-de](https://huggingface.co/tan-ht/naist-iwslt2025/tree/main/en-de) | ~4.43 GB |
| English → Chinese | [en-zh](https://huggingface.co/tan-ht/naist-iwslt2025/tree/main/en-zh) | ~4.43 GB |
| English → Japanese | [en-ja](https://huggingface.co/tan-ht/naist-iwslt2025/tree/main/en-ja) | ~4.43 GB |

The released models use **[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)**
and **[Whisper large-v3](https://huggingface.co/openai/whisper-large-v3)**,
with LoRA rank 8. Each language directory contains:

```text
en-de/
├── adapter_config.json        # PEFT adapter configuration
├── adapter_model.safetensors  # Adapter weights and saved embedding layers
├── checkpoint.pt             # Custom components and Fairseq configuration
└── README.md
```

The compact release does not bundle complete Qwen or Whisper base models and
omits optimizer state.
Use the entire language directory as `CKPT_DE`, `CKPT_ZH`, or `CKPT_JA`;
pointing only to its `checkpoint.pt` bypasses adapter loading.

## Configuration

Entry-point scripts under `script/` load `script/path.sh` and detect the
repository root automatically. Override paths with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_ROOT` | `<repo>/data` | Prepared datasets and evaluation inputs |
| `CKPT_DE` | `<repo>/checkpoints/en-de/checkpoint.pt` | German full checkpoint or adapter directory |
| `CKPT_ZH` | `<repo>/checkpoints/en-zh/checkpoint.pt` | Chinese full checkpoint or adapter directory |
| `CKPT_JA` | `<repo>/checkpoints/en-ja/checkpoint.pt` | Japanese full checkpoint or adapter directory |
| `LLM_NAME` | `qwen` | LLM implementation selected by the model |
| `LLM_PATH` | `Qwen/Qwen2.5-7B` | Hugging Face model ID or local path |

The default checkpoint paths target full Fairseq checkpoints. For the compact
Hugging Face release, use the directory overrides in Quick Start. For example,
with models and data stored elsewhere:

```bash
export DATA_ROOT=/datasets/iwslt2025
export CKPT_DE=/models/iwslt2025/en-de
```

## Data layout

The model loader needs a Fairseq speech-to-text data directory with a
`config.yaml` and the vocabulary files referenced by that configuration.
Training additionally uses split TSV manifests. Data preparation utilities
are under `iwslt/data_utils/`; input corpora are not redistributed.

Local-agreement evaluation expects:

```text
${DATA_ROOT}/en-<target-language>/
├── config.yaml
├── <vocabulary files>
└── simul_eval_acl/
    ├── source.txt            # One audio path per line
    └── target.txt            # One reference translation per line
```

Streaming evaluation expects:

```text
${DATA_ROOT}/
├── config.yaml
├── <vocabulary files>
├── src.txt                   # One audio path per line
└── tgt.txt                   # One reference translation per line
```

Audio paths must resolve on the machine running inference, and source and
reference lists must have matching line counts.

## Inference

Run these commands from the repository root with `.venv` activated and the
checkpoint directory overrides from Quick Start set.

### Local-agreement simultaneous inference

Evaluate English → German on GPU `0` across source chunk sizes of
500, 750, 1000, 1500, 2000, 2500, and 3000 ms:

```bash
bash script/simuleval_LA.sh en de 0
```

The arguments are source language, target language (`de`, `zh`, or `ja`),
and GPU index.

For a short smoke test, set `END_INDEX` to the number of examples to process:

```bash
END_INDEX=10 bash script/simuleval_LA.sh en de 0
```

Without `END_INDEX`, the complete input is evaluated.

### Streaming inference with SHAS segmentation

The streaming pipeline uses SHAS speech segmentation followed by
local-agreement decoding. The following settings reproduce the submitted
latency configurations:

| Language | Latency | Chunk size | Step length | SHAS threshold | End bias |
| --- | --- | ---: | ---: | ---: | ---: |
| German | Low | 124 | 0.62 | 0.2 | 400 |
| German | High | 250 | 2.0 | 0.2 | 400 |
| Chinese | Low | 170 | 0.85 | 0.4 | 400 |
| Chinese | High | 250 | 2.0 | 0.4 | 400 |
| Japanese | Low | 250 | 1.5 | 0.4 | 400 |
| Japanese | High | 250 | 2.5 | 0.4 | 400 |

For example, run the German low-latency configuration on GPU `0`:

```bash
bash script/streaming_shas_final.sh de low 0
```

The arguments are target language (`de`, `zh`, or `ja`), latency setting
(`low` or `high`), and GPU index. Prepare the streaming data layout above
before running; results are written under `exp_results/stream_final/`.

The included SHAS segmentation checkpoint is
`iwslt/speech_segmentation/en_shas_model.pt`. Translation checkpoints are
read from the `CKPT_*` environment variables.

## Model training

`script/train.sh` is the consolidated training entry point. Its main settings
can be overridden without editing the script:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TGT_LANG=zh \
TRAIN_DATA_ROOT=/datasets/covost_en-zh \
TRAIN_SPLIT=train_covost_en-zh \
VALID_SPLIT=dev_acl6060_en_zh \
LLM_NAME=qwen \
LLM_PATH=Qwen/Qwen2.5-7B \
WANDB_MODE=offline \
bash script/train.sh
```

By default, checkpoints are written below `checkpoints/`. Set
`WANDB_MODE=online` only when you intend to upload experiment metadata.

### Export a PEFT adapter

A full Fairseq checkpoint can be converted into a compact checkpoint plus a
PEFT adapter:

```bash
python iwslt/ckpt_utils/convert_to_peft.py \
  --input checkpoints/avg_best10.pt \
  --output checkpoints/best_adapter \
  --data-root /path/to/fairseq-manifest
```

The export utility extracts base-model weights from the Fairseq model state.
Training state may still be present in the exported `checkpoint.pt`; the
published Hugging Face release additionally removes optimizer and training
history fields.

## Docker

<details>
<summary>Build and run with the locked environment</summary>

Download the models and prepare data on the host first. The image installs the
locked environment; mount your data, checkpoints, and output directory at run
time:

```bash
docker build -t iwslt2025 .
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/checkpoints:/checkpoints:ro" \
  -v "$PWD/exp_results:/app/exp_results" \
  -e DATA_ROOT=/data \
  -e CKPT_DE=/checkpoints/en-de \
  -e CKPT_ZH=/checkpoints/en-zh \
  -e CKPT_JA=/checkpoints/en-ja \
  iwslt2025
```

The checkpoint paths above use the compact Hugging Face release. The container
also needs access to the Qwen and Whisper base models on first use.

</details>

## Release checks

Run the lightweight checks used by CI before committing:

```bash
PYTHON=.venv/bin/python bash script/check_release.sh
uv lock --check --offline
```

These checks validate whitespace, project shell syntax, Python syntax, and lock
file consistency. Full training and inference tests require GPU hardware,
datasets, and model checkpoints.

## Third-party components

Bundled third-party directories and assets retain their applicable upstream
terms. Source links and notices are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{tan2025naist,
  title={NAIST simultaneous speech translation system for IWSLT 2025},
  author={Tan, Haotian and Widiaputri, Ruhiyah Faradishi and Saragih, Jan Meyer and Ko, Yuka and Sudoh, Katsuhito and Nakamura, Satoshi and Sakti, Sakriani},
  booktitle={Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT 2025)},
  pages={369--378},
  year={2025}
}
```
