#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <checkpoint-dir> <output-dir> <number-of-best-checkpoints>" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_dir="$1"
output_dir="$2"
checkpoint_count="$3"
output_name="avg_best${checkpoint_count}.pt"

mkdir -p "${output_dir}"
python3 "${script_dir}/average_checkpoints.py" \
    --inputs "${checkpoint_dir}" \
    --output "${output_dir}/${output_name}" \
    --num-best-checkpoints "${checkpoint_count}"
