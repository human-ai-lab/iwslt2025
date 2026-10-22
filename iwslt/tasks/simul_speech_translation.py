from fairseq.tasks import FairseqTask, register_task
import logging
import torch
from pathlib import Path
from fairseq.data import Dictionary
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDatasetCreator,
    SpeechToTextDataset,
    S2TDataConfig
    )
import os
import json

logger = logging.getLogger(__name__)

@register_task("llm_sst_task")
class LlmSSTTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument('--max-source-positions', type=int, default=1e6, help='Maximum input length')
        parser.add_argument('--max-target-positions', type=int, default=1024, help='Maximum output length')

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.args = args
        self.valid_num = 0
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()
        self.model = None
        self.audio_input = True  # Ensure src_tokens on cpu in data loading

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a given dataset split (train/valid/test)."""
        is_train_split = split.startswith("train")
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            root=self.args.data,
            cfg=self.data_cfg,
            splits=split,
            tgt_dict=self.tgt_dict,
            pre_tokenizer=None,
            bpe_tokenizer=self.model.llama_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
            multitask=None,
        )
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        dict_path = Path(args.data) / data_cfg.vocab_filename
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")

        tgt_dict = Dictionary.load(dict_path.as_posix(), add_special_symbols=False)
        # set bos_index, eos_index, pad_index, unk_index
        if (args.llm_name == "llama"):
            tgt_dict.bos_index = tgt_dict.index('<|begin_of_text|>')
            tgt_dict.eos_index = tgt_dict.index('<|eot_id|>')
        else:
            generation_config_path = os.path.join(args.user_dir, 'tokenizer/llm_tokenizer/', args.llm_name, f'{args.llm_name}_generation_config.json')
            with open(generation_config_path) as f:
                generation_config = json.load(f)

            tgt_dict.bos_index = generation_config['bos_token_id']
            tgt_dict.eos_index = generation_config['eos_token_id']
            tgt_dict.pad_index = tgt_dict.eos_index
        tgt_dict.unk_index = tgt_dict.index("<unk>")
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    @property
    def target_dictionary(self):
        return self.tgt_dict
    
    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.args.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id
    
    def max_positions(self):
        """
        Return the maximum source and target positions.
        """
        return (self.args.max_source_positions, self.args.max_target_positions)
    
    def build_model(self, args):
        """
        Build the SalmonSST model.
        """
        model = super().build_model(args)
        self.model = model
        
        return self.model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        """
        Override to train using audio features and text.
        """
        self.valid_num = 0
        loss, sample_size, logging_output = super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                loss, sample_size, logging_output = criterion(model, sample)
        # BLEU
        if self.valid_num < 500:
            hyps, refs = self.inference_(model, sample)
            if self.valid_num < 5:
                logger.info("example hypothesis: " + hyps[0] + "\n")
                logger.info("example reference: " + refs[0] + "\n")
            logging_output["hyps"] = hyps
            logging_output["refs"] = refs
            logging_output['lang'] = sample['net_input']['tgt_langs'][0]
            self.valid_num += 1

        return loss, sample_size, logging_output

    def inference_(self, model, sample):
        targets = sample['net_input']['targets']
        src_tokens = sample['net_input']['src_tokens']
        tgt_langs = sample['net_input']['tgt_langs']
        hypos = model.generate(src_tokens, tgt_langs=tgt_langs, args=self.args)
        hypos = model.decode_fn(hypos)
        hypos = [hypo.replace("Translation:", "").strip() for hypo in hypos]
        refs = model.llama_tokenizer.batch_decode(targets, skip_special_tokens=True)
    
        return hypos, refs

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
