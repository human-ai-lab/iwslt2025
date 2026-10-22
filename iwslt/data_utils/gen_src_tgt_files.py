import pandas as pd
import os

""" Generate src.txt and tgt.txt files for simultaneous evaluation """

data_root_path = "../../data/covost/en-ja/"
test_tsv_file_path = os.path.join(data_root_path, "test_acl6060_ja.tsv")
out_root = os.path.join(data_root_path, "simul_eval")
os.makedirs(out_root, exist_ok=True)
out_audio_path = os.path.join(out_root, "source.txt")
out_text_path = os.path.join(out_root, "target.txt")

df = pd.read_csv(test_tsv_file_path, sep='\t')
df["audio"] = df["audio"].replace(r'^\s*$', pd.NA, regex=True)
df["tgt_text"] = df["tgt_text"].replace(r'^\s*$', pd.NA, regex=True)
df_clean = df.dropna(subset=["audio", "tgt_text"]).reset_index(drop=True)
assert len(df_clean["audio"]) == len(df_clean["tgt_text"]), "ERROR: 清洗后行数仍不一致！"
has_newline = df_clean["tgt_text"].str.contains("\n").any()
print("是否存在含换行符的文本:", has_newline)  # 若输出True则确认问题
if has_newline:
    print("含换行符的文本如下:")
    print(df_clean[df_clean["tgt_text"].str.contains("\n")])
    df_clean["tgt_text"] = df_clean["tgt_text"].str.replace("\n", " ").str.strip()
    has_newline = df_clean["tgt_text"].str.contains("\n").any()
    if has_newline:
        print("ERROR: 仍存在含换行符的文本！")
        exit(1)


with open(out_audio_path, 'w', encoding='utf-8') as f:
    for audio_path in df_clean['audio']:
        f.write(f"{audio_path}\n")

with open(out_text_path, 'w', encoding='utf-8') as f:
    for text in df_clean['tgt_text']:
        text.replace("\n", " ").strip()
        f.write(f"{text}\n")

print(f"Data saved to {out_root}")



