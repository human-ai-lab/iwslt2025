# NAIST Simultaneous Speech Translation System for IWSLT 2025

This repository contains the NAIST submission to the IWSLT 2025
Simultaneous Speech Translation Track. It supports simultaneous translation
from English into German (`en-de`), Chinese (`en-zh`), and Japanese
(`en-ja`).

The system is described in the
[IWSLT 2025 system paper](https://aclanthology.org/2025.iwslt-1.39/).

## Requirements

The tested environment is Linux on x86-64 with:

- Python 3.10
- an NVIDIA GPU and a driver compatible with the CUDA 12 dependencies in
  `uv.lock`
- sufficient storage for Whisper, the selected LLM, datasets, and checkpoints
- network access to Hugging Face for the first model download

Translation checkpoints and datasets are not included. You must obtain or train
them separately and comply with their respective terms. Some Hugging Face
models, including Llama-family models, may require account approval.

## Installation

Install [uv](https://docs.astral.sh/uv/), then create the locked environment:

```bash
git clone https://github.com/ha3ci-lab/iwslt2025.git
cd iwslt2025
uv sync --frozen
source .venv/bin/activate
```

The repository vendors the Fairseq and SimulEval versions used by the system.
Do not install different global copies over the locked environment.

### Docker

The image installs the same locked environment and does not copy local data,
checkpoints, experiment outputs, virtual environments, or Git metadata:

```bash
docker build -t iwslt2025 .
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/checkpoints:/checkpoints:ro" \
  -v "$PWD/exp_results:/app/exp_results" \
  -e DATA_ROOT=/data \
  -e CKPT_DE=/checkpoints/en-de/checkpoint.pt \
  -e CKPT_ZH=/checkpoints/en-zh/checkpoint.pt \
  -e CKPT_JA=/checkpoints/en-ja/checkpoint.pt \
  iwslt2025
```

## Configuration

Every entry-point script loads `script/path.sh`. The repository root is
detected automatically. Override data and checkpoint locations with environment
variables instead of committing machine-specific paths:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_ROOT` | `<repo>/data` | Prepared datasets and evaluation inputs |
| `CKPT_DE` | `<repo>/checkpoints/en-de/checkpoint.pt` | German checkpoint |
| `CKPT_ZH` | `<repo>/checkpoints/en-zh/checkpoint.pt` | Chinese checkpoint |
| `CKPT_JA` | `<repo>/checkpoints/en-ja/checkpoint.pt` | Japanese checkpoint |
| `LLM_NAME` | `qwen` | LLM implementation selected by the model |
| `LLM_PATH` | `Qwen/Qwen2.5-7B` | Hugging Face model ID or local path |

For example:

```bash
export DATA_ROOT=/datasets/iwslt2025
export CKPT_DE=/models/iwslt2025/en-de/checkpoint.pt
```

## Data layout

Training uses a Fairseq speech-to-text manifest directory containing a
`config.yaml`, vocabulary, and split TSV files. Data preparation utilities are
under `iwslt/data_utils/`; their input corpora are not redistributed.

Local-agreement evaluation expects:

```text
${DATA_ROOT}/en-<target-language>/simul_eval_acl/
├── source.txt
└── target.txt
```

Streaming evaluation expects:

```text
${DATA_ROOT}/
├── src.txt
└── tgt.txt
```

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

A full Fairseq checkpoint can be converted into a compact checkpoint plus a
PEFT adapter:

```bash
python iwslt/ckpt_utils/convert_to_peft.py \
  --input checkpoints/avg_best10.pt \
  --output checkpoints/best_adapter \
  --data-root /path/to/fairseq-manifest
```

## Inference

### Local-agreement simultaneous inference

The local-agreement entry point evaluates several source chunk sizes:

```bash
bash script/simuleval_LA.sh en <de|zh|ja> <device>
```

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

Run a configuration with:

```bash
bash script/streaming_shas_final.sh <de|zh|ja> <low|high> <device>
```

The included SHAS segmentation checkpoint is
`iwslt/speech_segmentation/en_shas_model.pt`. Translation checkpoints are
read from the `CKPT_*` environment variables.

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
