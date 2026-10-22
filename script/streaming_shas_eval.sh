#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/path.sh"

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <de|zh|ja> <end-bias> <shas-threshold>" >&2
    exit 1
fi

export MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-${IWSLT25_ROOT}/mwerSegmenter}"

tgt_lang="$1"
end_bias="$2"
shas_threshold="$3"
log_root="${IWSLT25_ROOT}/exp_results/stream_final/${tgt_lang}/qwen_LA-shas"
reference_root="${DATA_ROOT}/en-${tgt_lang}/stream_eval_acl"

case "${tgt_lang}" in
    de) tokenizer=13a; unit=word ;;
    zh) tokenizer=zh; unit=char ;;
    ja) tokenizer=ja-mecab; unit=char ;;
    *)
        echo "Unsupported target language: ${tgt_lang}" >&2
        exit 1
        ;;
esac

shopt -s nullglob
logs=("${log_root}"/end-bias_"${end_bias}"_shas-thred_"${shas_threshold}"_cs-*/instances.log)
if [[ ${#logs[@]} -eq 0 ]]; then
    echo "No matching logs found under ${log_root}" >&2
    exit 1
fi

for log_path in "${logs[@]}"; do
    output_path="$(dirname "${log_path}")/stream_scores.tsv"
    python3 "${IWSLT25_ROOT}/iwslt/exp_utils/stream_laal.py" \
        --simuleval-instances "${log_path}" \
        --reference "${reference_root}/tgt_segments.txt" \
        --audio-yaml "${reference_root}/acl6060.yaml" \
        --sacrebleu-tokenizer "${tokenizer}" \
        --latency-unit "${unit}" >"${output_path}"
done
