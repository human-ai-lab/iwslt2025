from transformers import AutoTokenizer
import sentencepiece as sp
import os
from tqdm import tqdm

llama_path = 'meta-llama/Llama-3.1-8B'
spm_path = "./spm_models"
save_path = './llama_tokenizer'

llama_tokenizer = AutoTokenizer.from_pretrained(llama_path)

for spm_file in os.listdir(spm_path):
    print(f"Merging with {spm_file}")
    spm_file = os.path.join(spm_path, spm_file)
    spm_model = sp.SentencePieceProcessor()
    spm_model.load(spm_file)
    for i in tqdm(range(spm_model.GetPieceSize()), desc="Merging SentencePiece model"):
        piece = spm_model.IdToPiece(i)
        if piece not in llama_tokenizer.vocab:
            llama_tokenizer.add_tokens(piece)

print("Size of the merged tokenizer:", llama_tokenizer.vocab_size)
llama_tokenizer.save_pretrained(save_path)