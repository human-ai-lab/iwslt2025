#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/path.sh"

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <source-language> <de|zh|ja> <device>" >&2
    exit 1
fi

src_lang="$1"
tgt_lang="$2"
device="$3"

case "${tgt_lang}" in
    de) checkpoint="${CKPT_DE}" ;;
    zh) checkpoint="${CKPT_ZH}" ;;
    ja) checkpoint="${CKPT_JA}" ;;
    *)
        echo "Unsupported target language: ${tgt_lang}" >&2
        exit 1
        ;;
esac

policy=local_agreement
chunk_sizes=(500 750 1000 1500 2000 2500 3000)
data_root="${DATA_ROOT}/${src_lang}-${tgt_lang}"
data_dir="${data_root}/simul_eval_acl"
agent="${IWSLT25_ROOT}/iwslt/agents/${policy}.py"
src_file="${data_dir}/source.txt"
tgt_file="${data_dir}/target.txt"
end_index="${END_INDEX:-}"
llm_name="${LLM_NAME:-qwen}"

for required_file in "${checkpoint}" "${agent}" "${src_file}" "${tgt_file}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

command -v simuleval >/dev/null || {
    echo "simuleval not found; run 'uv sync' and activate .venv first." >&2
    exit 1
}

base_cmd=(
    simuleval
    --agent "${agent}"
    --user-dir "${IWSLT25_ROOT}/iwslt"
    --data-root "${data_root}"
    --source "${src_file}"
    --target "${tgt_file}"
    --target-lang "${tgt_lang}"
    --model-path "${checkpoint}"
    --device "${device}"
    --fp16
    --computation-aware
    --llm-name "${llm_name}"
)

if [[ -n "${end_index}" ]]; then
    base_cmd+=(--end-index "${end_index}")
fi

case "${tgt_lang}" in
    zh) base_cmd+=(--eval-latency-unit char --sacrebleu-tokenizer zh) ;;
    ja) base_cmd+=(--eval-latency-unit char --sacrebleu-tokenizer ja-mecab) ;;
    de) base_cmd+=(--eval-latency-unit word --sacrebleu-tokenizer 13a) ;;
esac

for chunk_size in "${chunk_sizes[@]}"; do
    output_dir="${IWSLT25_ROOT}/exp_results/results_simul/${tgt_lang}/LA_chunk-size_${chunk_size}"
    run_cmd=("${base_cmd[@]}" --source-segment-size "${chunk_size}" --output "${output_dir}")
    printf 'Running:'
    printf ' %q' "${run_cmd[@]}"
    printf '\n'
    "${run_cmd[@]}"
done
