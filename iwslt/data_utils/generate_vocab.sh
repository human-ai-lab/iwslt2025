DATA_ROOT=""
OUT_ROOT=" "
export PROJECT_PATH="$PWD/../.."

python ./expand_llama_vocab.py \
    --data-root ${DATA_ROOT} \
    --output-root ${OUT_ROOT}  \
    --vocab-type bpe