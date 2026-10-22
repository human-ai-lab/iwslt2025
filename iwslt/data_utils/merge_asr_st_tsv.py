import pandas as pd

""" Incorporate ASR transcriptions into the ST training data. """

class MergeAsrStTsv:
    def __init__(self, st_tsv_path, asr_tsv_path, out_tsv_path):
        self.st_tsv_path = st_tsv_path
        self.asr_tsv_path = asr_tsv_path
        self.out_tsv_path = out_tsv_path

    def merge(self):
        st_df = pd.read_csv(self.st_tsv_path, sep="\t", encoding="utf-8")
        asr_df = pd.read_csv(self.asr_tsv_path, sep="\t", encoding="utf-8")

        merged_df = pd.merge(
            st_df[["id", "audio", "n_frames", "tgt_text", "speaker", "tgt_lang"]],
            asr_df[["id", "tgt_text"]],
            on="id",
            how="left",
            suffixes=("", "_asr")
        )

        # Rename columns and reorder
        merged_df = merged_df.rename(columns={"tgt_text_asr": "src_text"})
        merged_df = merged_df[["id", "audio", "n_frames", "tgt_text", "src_text", "speaker", "tgt_lang"]]
        merged_df["speaker"] = "None"
        # Save results
        merged_df.to_csv(self.out_tsv_path, sep="\t", index=False, encoding="utf-8")


if __name__ == "__main__":
    st_tsv_path = "../../data/covost_europarl/fixed_train_st_filtered.tsv"
    asr_tsv_path = "../../data/covost_europarl/fixed_train_asr_filtered.tsv"
    out_tsv_path = "../../data/covost_europarl/train_st_asr_covost.tsv"
    merge_asr_st_tsv = MergeAsrStTsv(st_tsv_path, asr_tsv_path, out_tsv_path)
    merge_asr_st_tsv.merge()