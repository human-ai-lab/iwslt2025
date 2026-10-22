#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/path.sh"

export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_NAME="${WANDB_NAME:-covost_en_zh_st_asr}"
export WANDB_MODE="${WANDB_MODE:-offline}"

IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
gpu_nums="${#visible_gpus[@]}"

# Data and model settings can be overridden with environment variables.
tgt_lang="${TGT_LANG:-zh}"
train_data_root="${TRAIN_DATA_ROOT:-${DATA_ROOT}/covost_en-zh}"
ckpt_save_path="${CHECKPOINT_DIR:-${IWSLT25_ROOT}/checkpoints/${WANDB_NAME}}"
train_split="${TRAIN_SPLIT:-train_covost_en-zh}"
valid_split="${VALID_SPLIT:-dev_acl6060_en_zh}"
user_dir="${IWSLT25_ROOT}/iwslt"
llm_name="${LLM_NAME:-qwen}"
llm_path="${LLM_PATH:-Qwen/Qwen2.5-7B}"

seed="${SEED:-0}"
chunk_size="${CHUNK_SIZE:-1000}"
max_tokens="${MAX_TOKENS:-600000}"
max_sentences="${MAX_SENTENCES:-15}"
max_tokens_valid="${MAX_TOKENS_VALID:-600000}"
max_source_len="${MAX_SOURCE_LEN:-400000}"
update_freq="${UPDATE_FREQ:-1}"
patience="${PATIENCE:-15}"
validate_interval_updates="${VALIDATE_INTERVAL_UPDATES:-1500}"
continue_from_ckpt="${CONTINUE_FROM_CKPT:-None}"

if [[ ! -d "${train_data_root}" ]]; then
    echo "Training data directory not found: ${train_data_root}" >&2
    echo "Set TRAIN_DATA_ROOT to a prepared Fairseq speech-to-text manifest directory." >&2
    exit 1
fi

command -v fairseq-train >/dev/null || {
    echo "fairseq-train not found; run 'uv sync' and activate .venv first." >&2
    exit 1
}

fairseq-train "${train_data_root}" \
    --user-dir "${user_dir}" \
    --task llm_sst_task \
    --arch llm_sst_pool_8B \
    --save-dir "${ckpt_save_path}" \
    --train-subset "${train_split}" \
    --valid-subset "${valid_split}" \
    --chunk-size "${chunk_size}" \
    --freeze-whisper \
    --prompt-path "${user_dir}/prompts/train_prompt.json" \
    --target-lang "${tgt_lang}" \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 2e-4 --lr-scheduler cosine --warmup-updates 3000 --warmup-init-lr 2e-06 --stop-min-lr 1e-6 \
    --weight-decay 0.05 --criterion dummy_criterion \
    --fp16 \
    --spec-augment \
    --max-tokens "${max_tokens}" --max-sentences "${max_sentences}" \
    --max-tokens-valid "${max_tokens_valid}" --update-freq "${update_freq}" \
    --max-source-positions "${max_source_len}" --max-target-positions 1024 \
    --keep-best-checkpoints 10 --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-interval-updates 0 --save-interval-updates "${validate_interval_updates}" --save-interval 10 \
    --no-last-checkpoints --continue-once "${continue_from_ckpt}" \
    --seed "${seed}" --fixed-validation-seed "${seed}" \
    --skip-invalid-size-inputs-valid-test \
    --max-update 100000 --max-epoch 30 --patience "${patience}" \
    --distributed-world-size "${gpu_nums}" --model-parallel-size 1 --ddp-backend pytorch_ddp \
    --offline \
    --num-workers 8 \
    --heartbeat-timeout -1 \
    --wandb-project llm_sst \
    --llm-name "${llm_name}" \
    --llm-path "${llm_path}"
