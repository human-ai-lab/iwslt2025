#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple
import pandas as pd
import torchaudio
import os
import sys
import soundfile as sf

project_path = os.getenv("PROJECT_PATH")
if project_path:
    sys.path.append(project_path)
else:
    print("Warning: PROJECT_PATH environment variable not set. Some imports might fail.")

try:
    from fairseq.examples.speech_to_text.data_utils import (
        create_zip,
        filter_manifest_df,
        gen_config_yaml,
        get_zip_manifest,
        load_df_from_tsv,
        save_df_to_tsv,
    )
    from torch import Tensor
    from torch.utils.data import Dataset
    from torch.hub import download_url_to_file as download_url
    from fairseq.data.audio.audio_utils import convert_waveform
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure fairseq is installed and PROJECT_PATH is set correctly.")
    sys.exit(1)

import tarfile
import zipfile
from tqdm import tqdm

log = logging.getLogger(__name__)

def extract_archive(from_path, to_path=None, overwrite=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)
    os.makedirs(to_path, exist_ok=True)
    if tarfile.is_tarfile(from_path):
        with tarfile.open(from_path, 'r:*') as tar:
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path, members, numeric_owner=numeric_owner)
            safe_extract(tar, path=to_path)
    elif zipfile.is_zipfile(from_path):
        with zipfile.ZipFile(from_path, 'r') as zf:
            zf.extractall(to_path)
    else:
        raise ValueError(f"Unsupported archive format: {from_path}")

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "src_text", "speaker", "tgt_lang"]

class CoVoST(Dataset):
    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )
    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]
    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr", "de", "es", "ca", "it", "ru", "zh-CN", "pt", "fa", "et", "mn",
            "nl", "tr", "ar", "sv-SE", "lv", "sl", "ta", "ja", "id", "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de", "tr", "fa", "sv-SE", "mn", "zh-CN", "cy", "ca", "sl", "et",
            "id", "ar", "ta", "lv", "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            target_language = "de" if source_language == "en" else "en"
        self.root: Path = Path(root)
        self.clips_root: Path = self.root.parent / "clips"

        cv_tsv_path = self.root / "validated.tsv"
        if not cv_tsv_path.is_file():
             cv_tsv_path_alt = self.root / "validated.txt"
             if cv_tsv_path_alt.is_file():
                 cv_tsv_path = cv_tsv_path_alt
             else:
                 raise FileNotFoundError(f"validated.tsv (or .txt) not found in {self.root}")

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        covost_archive = self.root / Path(covost_url).name
        if not covost_archive.is_file():
            print(f"Downloading CoVoST TSV from {covost_url}...")
            download_url(covost_url, os.path.join(self.root, os.path.basename(covost_url)), hash_prefix=None)
        print(f"Extracting CoVoST TSV archive {covost_archive}...")
        extract_archive(covost_archive.as_posix())
        covost_tsv_file = Path(covost_url).name.replace(".tar.gz", "")
        covost_tsv_path = self.root / covost_tsv_file
        if not covost_tsv_path.is_file():
             covost_tsv_path_alt = self.root / (covost_tsv_file + ".txt")
             if covost_tsv_path_alt.is_file():
                 covost_tsv_path = covost_tsv_path_alt
             else:
                 raise FileNotFoundError(f"CoVoST TSV file {covost_tsv_file} (or .txt) not found in {self.root} after extraction.")

        print(f"Loading Common Voice TSV from {cv_tsv_path}...")
        cv_tsv = load_df_from_tsv(cv_tsv_path)
        print(f"Loading CoVoST TSV from {covost_tsv_path}...")
        covost_tsv = load_df_from_tsv(covost_tsv_path)
        print("Merging TSVs...")
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        print(f"Filtering for split: {split}...")
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        print("Validating audio files...")
        for e in tqdm(data):
            try:
                path = self.clips_root / e["path"]
                if not path.is_file():
                     print(f"Warning: Audio file not found: {path}")
                     continue
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except Exception as e_info:
                print(f"Warning: Skipping invalid audio file {path}: {e_info}")
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        data = self.data[n]
        path = self.clips_root / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)

def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")

    audio_root = root / "audio"
    audio_root.mkdir(exist_ok=True)

    print("Processing audio files...")
    for split in CoVoST.SPLITS:
        print(f"Fetching split {split}...")
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        print("Converting and saving audio files...")
        for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
            output_path = audio_root / f"{utt_id}.flac"
            if output_path.exists():
                continue

            tgt_sample_rate = 16_000
            waveform_mono_16k, _ = convert_waveform(
                waveform, sample_rate, to_mono=True, to_sample_rate=tgt_sample_rate
            )
            sf.write(
                output_path.as_posix(),
                waveform_mono_16k.T.numpy(),
                tgt_sample_rate
            )

    zip_path = root / "audio.zip"
    print("ZIPing audio files...")
    create_zip(audio_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path, is_audio=True)

    print("Generating manifest...")
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"

    for split in CoVoST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        print(f"Creating manifest for split: {split}")
        for _, _, src_utt, tgt_utt, _speaker_id_unused, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths.get(utt_id, ""))
            manifest["n_frames"].append(audio_lengths.get(utt_id, 0))
            manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
            manifest["src_text"].append(src_utt)
            manifest["speaker"].append("None")
            target_language_entry = args.tgt_lang if args.tgt_lang is not None else args.src_lang
            manifest["tgt_lang"].append(target_language_entry)

        is_train_split = split.startswith("train")
        df = pd.DataFrame.from_dict(manifest)
        df = df[MANIFEST_COLUMNS]
        df = filter_manifest_df(df, is_train_split=is_train_split, max_n_frames=480000)
        save_df_to_tsv(df, root / f"{split}_{task}.tsv")

    print("Cleaning up temporary audio directory...")
    shutil.rmtree(audio_root)
    print("Processing complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str, help="Target language for speech translation. If omitted, ASR is assumed.")
    parser.add_argument("--use-audio-input", action="store_true", default=True, help="Use raw audio as input (default: True).")

    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"Error: Data root directory '{args.data_root}' does not exist.")
        return

    expected_src_dir = os.path.join(args.data_root, args.src_lang)
    if not os.path.isdir(expected_src_dir):
        print(f"Error: Expected source language directory '{expected_src_dir}' does not exist.")
        print("Please ensure the Common Voice data for the source language is correctly placed.")
        return

    validated_path = os.path.join(expected_src_dir, "validated.tsv")
    validated_path_alt = os.path.join(expected_src_dir, "validated.txt")
    if not (os.path.isfile(validated_path) or os.path.isfile(validated_path_alt)):
        print(f"Error: Neither 'validated.tsv' nor 'validated.txt' found in '{expected_src_dir}'.")
        print("Please ensure the Common Voice data is correctly extracted.")
        return

    try:
        process(args)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise

if __name__ == "__main__":
    main()