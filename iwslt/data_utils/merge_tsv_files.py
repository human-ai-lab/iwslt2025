# Merge two tsv files
import pandas as pd

def merge_tsv_files(file1, file2, output_file):
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(output_file, sep='\t', index=False)

if __name__ == '__main__':
    file1 = "file1.tsv"
    file2 = "file2.tsv"
    output = "merged_output.tsv"
    merge_tsv_files(file1, file2, output)
