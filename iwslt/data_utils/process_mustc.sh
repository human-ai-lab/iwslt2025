#!/bin/bash
DATA_ROOT=data/must-c1.0
OUT_ROOT=iwslt2025/data/must-c1.0
export PROJECT_PATH=".../iwslt2025"

python ./prep_mustc_data.py \
    --data-root ${DATA_ROOT} \
    --task "st" \
    --output-root ${OUT_ROOT}