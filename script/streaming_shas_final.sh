#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/path.sh"
ulimit -n 65535 2>/dev/null || true

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <de|zh|ja> <low|high> <device>" >&2
    exit 1
fi

tgt_lang="$1"
latency="$2"
device="$3"
end_bias=400
attention_threshold=0.1
max_segment_length=15
hypothesis_policy=local_agreement
shas_checkpoint="${IWSLT25_ROOT}/iwslt/speech_segmentation/en_shas_model.pt"
output_segments=0

case "${tgt_lang}:${latency}" in
    de:low) chunk_size=124; step_length=0.62; shas_threshold=0.2 ;;
    de:high) chunk_size=250; step_length=2.0; shas_threshold=0.2 ;;
    zh:low) chunk_size=170; step_length=0.85; shas_threshold=0.4 ;;
    zh:high) chunk_size=250; step_length=2.0; shas_threshold=0.4 ;;
    ja:low) chunk_size=250; step_length=1.5; shas_threshold=0.4 ;;
    ja:high) chunk_size=250; step_length=2.5; shas_threshold=0.4 ;;
    *)
        echo "Unsupported configuration: language=${tgt_lang}, latency=${latency}" >&2
        echo "Usage: $0 <de|zh|ja> <low|high> <device>" >&2
        exit 1
        ;;
esac

case "${tgt_lang}" in
    de) checkpoint="${CKPT_DE}" ;;
    zh) checkpoint="${CKPT_ZH}" ;;
    ja) checkpoint="${CKPT_JA}" ;;
esac

output_dir="${IWSLT25_ROOT}/exp_results/stream_final/${tgt_lang}/${tgt_lang}_${latency}_latency"
start_time="${SECONDS}"

bash "${IWSLT25_ROOT}/script/streaming_shas_run.sh" \
    "${step_length}" \
    "${chunk_size}" \
    "${attention_threshold}" \
    "${output_dir}" \
    0 \
    "${tgt_lang}" \
    "${checkpoint}" \
    "${device}" \
    "${hypothesis_policy}" \
    "${shas_checkpoint}" \
    "${shas_threshold}" \
    "${max_segment_length}" \
    "${end_bias}" \
    "${output_segments}"

elapsed_time=$((SECONDS - start_time))
echo "Elapsed time: $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds"
