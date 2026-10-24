#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/path.sh"

if [[ $# -ne 14 ]]; then
    echo "Usage: $0 <step-size> <chunk-size> <attention-threshold> <output-dir> <offline:0|1> <target-lang> <checkpoint> <device> <hypothesis-policy> <shas-checkpoint> <shas-threshold> <max-segment-length> <end-bias> <output-segments:0|1>" >&2
    exit 1
fi

step_size="$1"
chunk_size="$2"
attn_threshold="$3"
output_dir="$4"
offline="$5"
tgt_lang="$6"
checkpoint="$7"
device="$8"
hypo_policy="$9"
shas_checkpoint="${10}"
shas_threshold="${11}"
max_segment_length="${12}"
end_bias="${13}"
output_segments="${14}"

src_file="${DATA_ROOT}/src.txt"
tgt_file="${DATA_ROOT}/tgt.txt"
agent="${IWSLT25_ROOT}/iwslt/agents/stream_shas.py"

checkpoint_files=("${checkpoint}")
if [[ -d "${checkpoint}" ]]; then
    checkpoint_files=("${checkpoint}/checkpoint.pt" "${checkpoint}/adapter_model.safetensors")
fi

for required_file in "${checkpoint_files[@]}" "${shas_checkpoint}" "${agent}" "${src_file}" "${tgt_file}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

mkdir -p "${output_dir}"

cmd=(
    simuleval
    --agent "${agent}"
    --user-dir "${IWSLT25_ROOT}/iwslt"
    --data-root "${DATA_ROOT}"
    --model-path "${checkpoint}"
    --source-segment-size "${chunk_size}"
    --source "${src_file}"
    --target "${tgt_file}"
    --target-lang "${tgt_lang}"
    --step-length "${step_size}"
    --output "${output_dir}"
    --device "${device}"
    --fp16
    --no-scoring
    --out-attn
    --attn-thred "${attn_threshold}"
    --hypo-policy "${hypo_policy}"
    --shas-ckpt "${shas_checkpoint}"
    --seg-thred "${shas_threshold}"
    --end-bias "${end_bias}"
    --max-segment-length "${max_segment_length}"
    --llm-name "${LLM_NAME:-qwen}"
)

[[ "${output_segments}" == 1 ]] && cmd+=(--out-segment)
[[ "${offline}" == 1 ]] && cmd+=(--offline)

case "${tgt_lang}" in
    zh) cmd+=(--eval-latency-unit char --sacrebleu-tokenizer zh) ;;
    ja) cmd+=(--eval-latency-unit char --sacrebleu-tokenizer ja-mecab) ;;
    de) cmd+=(--eval-latency-unit word --sacrebleu-tokenizer 13a) ;;
    *)
        echo "Unsupported target language: ${tgt_lang}" >&2
        exit 1
        ;;
esac

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
