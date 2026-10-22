#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../script/path.sh"

shas_root="${IWSLT25_ROOT}/SHAS"
yaml_file="${SHAS_YAML:-${IWSLT25_ROOT}/data/shas/acl6060.yaml}"
wav_dir="${WAV_ROOT:?Set WAV_ROOT to the directory containing source WAV files}"
output_dir="${SHAS_OUTPUT_DIR:-${IWSLT25_ROOT}/data/shas}"

python3 "${shas_root}/src/data_prep/prepare_dataset_for_segmentation.py" \
    -y "${yaml_file}" \
    -w "${wav_dir}" \
    -o "${output_dir}"
