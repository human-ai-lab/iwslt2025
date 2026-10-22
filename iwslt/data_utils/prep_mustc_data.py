#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
The code is modified from fairseq.
Prepare MuST-C data using Whisper's feature extractor
'''
import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import soundfile as sf
import sys
project_path = os.getenv("PROJECT_PATH")
sys.path.append(project_path)
from fairseq.examples.speech_to_text.data_utils import (
    create_zip,
    filter_manifest_df,
    gen_config_yaml,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from tqdm import tqdm

from fairseq.data.audio.audio_utils import get_waveform, convert_waveform

log = logging.getLogger(__name__)
WhisperProcessor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru", "zh", "ja"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(
            self, n: int
    ) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, \
            utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    for lang in MUSTC.LANGUAGES:
        cur_root = root / f"en-{lang}"
        out_path = Path(args.output_root).absolute() / f"en-{lang}"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        # Extract features
        audio_root = out_path / ("flac" if args.use_audio_input else "fbank128")
        audio_root.mkdir(exist_ok=True)
        for split in MUSTC.SPLITS:
            print(f"Fetching split {split}...")
            dataset = MUSTC(root.as_posix(), lang, split)
            if args.use_audio_input:
                print("Converting audios...")
                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    if (audio_root / f"{utt_id}.flac").exists():
                        continue
                    tgt_sample_rate = 16_000
                    _wavform, _ = convert_waveform(
                        waveform, sample_rate, to_mono=True,
                        to_sample_rate=tgt_sample_rate
                    )
                    sf.write(
                        (audio_root / f"{utt_id}.flac").as_posix(),
                        _wavform.T.numpy(), tgt_sample_rate
                    )
            else:
                raise NotImplementedError("Feature extraction is not implemented yet.")

        # Pack features into ZIP
        zip_path = out_path / f"{audio_root.name}.zip"
        print("ZIPing audios/features...")
        create_zip(audio_root, zip_path)
        print("Fetching ZIP manifest...")
        audio_paths, audio_lengths = get_zip_manifest(
            zip_path,
            is_audio=args.use_audio_input,
        )
        # Generate TSV manifest
        print("Generating manifest...")
        train_text = []
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
            for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                manifest["audio"].append(audio_paths[utt_id])
                manifest["n_frames"].append(audio_lengths[utt_id])
                manifest["tgt_text"].append(
                    src_utt if args.task == "asr" else tgt_utt
                )
                manifest["speaker"].append(speaker_id)
            if is_train_split:
                train_text.extend(manifest["tgt_text"])
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split, max_n_frames=480000)  # Filter out samples longer than 30 seconds
            save_df_to_tsv(df, out_path / f"{split}_{args.task}.tsv")

        # Clean up
        shutil.rmtree(audio_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument(
        "--cmvn-type", default="utterance",
        choices=["global", "utterance"],
        help="The type of cepstral mean and variance normalization"
    )
    parser.add_argument(
        "--gcmvn-max-num", default=150000, type=int,
        help="Maximum number of sentences to use to estimate global mean and "
             "variance"
        )
    parser.add_argument("--use-audio-input", type=bool, default=True)
    args, unknown_args = parser.parse_known_args()
    process(args)


if __name__ == "__main__":
    main()
