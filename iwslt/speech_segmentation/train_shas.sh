#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../script/path.sh"

shas_root="${IWSLT25_ROOT}/SHAS"
segmentation_data_root="${SHAS_DATA_ROOT:-${IWSLT25_ROOT}/data/shas}"
results_root="${SHAS_RESULTS_ROOT:-${IWSLT25_ROOT}/iwslt/speech_segmentation}"
experiment_name="${SHAS_EXPERIMENT_NAME:-en_shas_model_tuned}"

python3 "${shas_root}/src/supervised_hybrid/train.py" \
    --datasets "${segmentation_data_root}/MUSTC/en-de" \
    --results_path "${results_root}/supervised_hybrid" \
    --model_name facebook/wav2vec2-xls-r-300m \
    --experiment_name "${experiment_name}" \
    --train_sets tst-COMMON \
    --eval_sets dev \
    --batch_size 14 \
    --learning_rate 2.5e-4 \
    --update_freq 20 \
    --max_epochs 8 \
    --classifier_n_transformer_layers 1 \
    --wav2vec_keep_layers 15
