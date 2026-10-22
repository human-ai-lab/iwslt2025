import pandas as pd

def modify_audio_path(data_path, output_path, src_prefix, tgt_prefix):
    df = pd.read_csv(data_path, sep='\t')
    source_prefix = src_prefix
    target_prefix = tgt_prefix
    df['audio'] = df['audio'].apply(lambda x: x.replace(source_prefix, target_prefix))
    df.to_csv(output_path, index=False, sep='\t')

if __name__ == '__main__':
    src_prefix = "src_prefix"
    tgt_prefix = "tgt_prefix"
    tsv_path = 'out.tsv'
    output_path = tsv_path
    modify_audio_path(tsv_path, output_path, src_prefix, tgt_prefix)