#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/path.sh"
ulimit -n 65535 2>/dev/null || true

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <de|zh|ja> <device>" >&2
    exit 1
fi

tgt_lang="$1"
device="$2"
chunk_size=250
hypo_policy=local_agreement
shas_checkpoint="${IWSLT25_ROOT}/iwslt/speech_segmentation/en_shas_model.pt"
max_segment_length=15
step_length=inf
output_segments=1
offline=1

case "${tgt_lang}" in
    de) checkpoint="${CKPT_DE}" ;;
    zh) checkpoint="${CKPT_ZH}" ;;
    ja) checkpoint="${CKPT_JA}" ;;
    *)
        echo "Unsupported target language: ${tgt_lang}" >&2
        exit 1
        ;;
esac

for end_bias in 400 800 1600; do
    for shas_threshold in 0.6 0.5 0.4 0.3 0.2 0.1; do
        out_dir="${IWSLT25_ROOT}/exp_results/stream_tune/shas/${tgt_lang}/qwen/end-bias_${end_bias}_shas-thred_${shas_threshold}"
        bash "${IWSLT25_ROOT}/script/streaming_shas_run.sh" \
            "${step_length}" "${chunk_size}" 0.1 "${out_dir}" "${offline}" \
            "${tgt_lang}" "${checkpoint}" "${device}" "${hypo_policy}" \
            "${shas_checkpoint}" "${shas_threshold}" "${max_segment_length}" \
            "${end_bias}" "${output_segments}"
    done
done
