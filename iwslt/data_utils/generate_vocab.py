from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, List
import sentencepiece as sp
import argparse
import logging
import os
import sys
from multiprocessing import cpu_count
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)

def gen_llama_vocab(
    input_path: Path, output_path_prefix: Path, model_type="bpe",
    vocab_size=10000, special_symbols: Optional[List[str]] = None
):
    """
    Note: To be consistent with the LLama setting, model_type="bpe", split_digits=True, byte_fallback=True, set max_sentencepiece_length a large number
    """
    UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
    BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
    EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
    PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
    # Train SentencePiece Model
    arguments = [
        f"--input={input_path.as_posix()}",
        f"--model_prefix={output_path_prefix.as_posix()}",
        f"--model_type={model_type}",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--num_threads={cpu_count()}",
        f"--split_digits=True",
        f"--byte_fallback=True",
        f"--max_sentence_length=24000",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
    ]
    if special_symbols is not None:
        _special_symbols = ",".join(special_symbols)
        arguments.append(f"--user_defined_symbols={_special_symbols}")
    sp.SentencePieceTrainer.Train(" ".join(arguments))
    # Export fairseq dictionary
    spm = sp.SentencePieceProcessor()
    spm.Load(output_path_prefix.as_posix() + ".model")
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix.as_posix() + ".txt", "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")

def process(args):
    tgt_langs = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru", "zh", "ja"]
    for lang in tgt_langs:
        data_dir = Path(args.data_root) / "en-{}".format(lang)
        # check existence of lang-dir
        if not data_dir.exists():
            print(f"Data directory {data_dir} does not exist, skipping...")
            continue
        txt_path = data_dir /"data"/ "train" / "txt" / f"train.{lang}"
        print(f"Processing file: {txt_path}")
        gen_llama_vocab(
            txt_path,
            Path(args.output_root) / "en-{}".format(lang) / "spm_{}_{}_st".format(args.vocab_type, args.vocab_size),
            model_type=args.vocab_type,
            vocab_size=args.vocab_size,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument(
        "--vocab-type",
        default="bpe",
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args, _ = parser.parse_known_args()
    process(args)

def test():
    sp_bpe = sp.SentencePieceProcessor()
    sp_bpe.load("spm_bpe_10000_st.model")
    test_text = "汽车和火车--我在上次幻灯片的时候谈到过， 但是我希望你们能够先思考一下。"
    # Get IDs for special tokens
    bos_id = sp_bpe.PieceToId("<s>")  # BOS token ID
    eos_id = sp_bpe.PieceToId("</s>")  # EOS token ID
    unk_id = sp_bpe.PieceToId("<unk>")  # UNK token ID
    pad_id = sp_bpe.PieceToId("<pad>")  # PAD token ID
    print(bos_id, eos_id, unk_id, pad_id)
    print(sp_bpe.encode_as_pieces(test_text))

if __name__ == "__main__":
    # main()
    test()
