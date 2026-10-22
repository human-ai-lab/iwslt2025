#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IWSLT25_ROOT="${IWSLT25_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-${IWSLT25_ROOT}/data}"
export CKPT_DE="${CKPT_DE:-${IWSLT25_ROOT}/checkpoints/en-de/checkpoint.pt}"
export CKPT_ZH="${CKPT_ZH:-${IWSLT25_ROOT}/checkpoints/en-zh/checkpoint.pt}"
export CKPT_JA="${CKPT_JA:-${IWSLT25_ROOT}/checkpoints/en-ja/checkpoint.pt}"

export PYTHONPATH="${IWSLT25_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
