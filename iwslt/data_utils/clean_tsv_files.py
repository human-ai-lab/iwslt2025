import pandas as pd
import os

""" Clean the tsv files """

class TSVAnomalyDetector:
    def __init__(self, tsv_path: str):
        self.tsv_path = tsv_path
        
    def tsv_load(self):
        try:
            self.df = pd.read_csv(self.tsv_path, sep="\t", encoding="utf-8")
            print(f"TSV文件{self.tsv_path}加载成功")
            return True
        except Exception as e:
            print(f"文件加载失败: {str(e)}")
            return False

    def check_unpexpected_strings(self, text: str):
        if any(char in text for char in ["\n", "\t"]) or ".wav" in text:
            return True
        else:
            return False

    def detect_issues(self):
        self.tsv_load()
        unclosed_lines, unclosed_contents = self.detect_and_fix_unclosed_quotes()
        # Check unexpected strings in the text
        for tgt_text in self.df["tgt_text"]:
            if self.check_unpexpected_strings(tgt_text):
                print(f"Unexpected strings in tgt_text: {tgt_text}")

        return unclosed_lines, unclosed_contents
    
    def detect_and_fix_unclosed_quotes(self):
        # Directly read the file using readlines()
        unclosed_lines = []
        unclosed_contents = []
        with open(self.tsv_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                try:
                    id_, audio_, n_frames, tgt_text, speaker, tgt_lang = line.strip().split('\t')
                except ValueError:
                    raise ValueError(f"Line {i} error: {line}")

                # Check tgt_lang
                if tgt_lang is None:
                    print(f"data id {id_} has no tgt_lang")

                # delete all the quotes in the tgt_text
                tgt_text = tgt_text.replace('"', '')
                lines[i] = '\t'.join([id_, audio_, n_frames, tgt_text, speaker, tgt_lang]) + '\n'

        # Write the fixed lines back to the file
        # Find dir path of the file
        dir_path = os.path.dirname(self.tsv_path)
        write_path = os.path.join(dir_path, "fixed_" + os.path.basename(self.tsv_path))
        with open(write_path, 'w', encoding='utf-8') as file:
            for line in lines:
                file.write(line)
        
        # Recheck the fixed file
        self.tsv_path = write_path
        self.tsv_load()
        unclosed_lines_fixed = []
        unclosed_contents_fixed = []
        for i, line in enumerate(self.df["tgt_text"]):
            if line.count('"') % 2 != 0:
                unclosed_lines_fixed.append(i)
                unclosed_contents_fixed.append(line)
        if len(unclosed_lines_fixed) > 0:
            print("Unclosed quotes still exist after fixing")
        else:
            print("Unclosed quotes fixed successfully")
        
        return unclosed_lines, unclosed_contents

if __name__ == "__main__":
    root_path = "../../data/covost_europarl"
    tsv_files = ['train_st_filtered.tsv', 'train_asr_filtered.tsv']
    for file_path in tsv_files:
        file_path = os.path.join(root_path, file_path)
        detector = TSVAnomalyDetector(file_path)
        anomaly = detector.detect_issues()






