import logging
import torch
import numpy as np
import math
import logging
from typing import Union
from simuleval.agents.actions import ReadAction
from simuleval.agents.states import AgentStates
from ..speech_segmentation.speech_segmenter import SpeechSegmenter, TranslateAction
from SHAS.src.supervised_hybrid.models import SegmentationFrameClassifer, prepare_wav2vec
from SHAS.src.supervised_hybrid.segment import Segment, trim

INPUT_SAMPLE_RATE= 16000
TARGET_SAMPLE_RATE = 49.95  # 50 (16000/320) wasnt exactly correct


class SHASSegmenter(SpeechSegmenter):
    '''
    Segment audio based on SHAS (https://github.com/mt-upc/SHAS)
    '''

    def __init__(self, args):
        self.device = "cuda:" + str(args.device)
        self.input_sr = INPUT_SAMPLE_RATE
        self.target_sr = TARGET_SAMPLE_RATE
        self.in_trg_ratio = self.input_sr / self.target_sr  # compression ratio
        self.trg_in_ratio = 1 / self.in_trg_ratio 
        self.max_segment_length = args.max_segment_length
        self.min_segment_length = args.min_segment_length
        self.seg_thred = args.seg_thred
        self.end_bias = getattr(args, "end_bias", None)
        self.not_strict = getattr(args, "not_strict", False)
        self.out_segment = getattr(args, "out_segment", False)
        assert self.end_bias is not None, "end_bias should be set"

        shas_ckpt = torch.load(args.shas_ckpt, map_location=self.device)
        # init wav2vec 2.0
        self.wav2vec_model = prepare_wav2vec(
            shas_ckpt["args"].model_name,
            shas_ckpt["args"].wav2vec_keep_layers,
            self.device,
            )
        # init segmentation frame classifier
        self.sfc_model = SegmentationFrameClassifer(
            d_model=1024,
            n_transformer_layers=shas_ckpt["args"].classifier_n_transformer_layers,
        ).to(self.device)
        self.sfc_model.load_state_dict(shas_ckpt["state_dict"])
        self.sfc_model.eval()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--shas-ckpt", type=str, help="Path to the SHAS model")
        parser.add_argument("--max-segment-length", default=18.0,
                            type=float, help="Maximum segment length in seconds")
        parser.add_argument("--min-segment-length", default=2.0,
                            type=float, help="Minimum segment length in seconds")
        parser.add_argument("--seg-thred", default=0.5,
                            type=float, help="Threshold for segmentation")
        parser.add_argument("--not-strict", action="store_true",
                            help="whether segments longer than max are allowed")
        parser.add_argument("--end-bias", type=int,
                            help="Segment tend to end early without bias. Default 0.1s")
        parser.add_argument("--out-segment", action="store_true",
                            help="whether to output the segment yaml file")
        
    def _inframes_to_outframes(self, x):
        # from input space to output space
        return np.round(x * self.trg_in_ratio).astype(int)
    
    def shas_segment(self, audio_: torch.Tensor):
        
        duration_outframes = self._inframes_to_outframes(audio_.size(1))
        in_mask = torch.ones(audio_.shape, dtype=torch.long).to(self.device)
        out_mask = torch.ones([duration_outframes], dtype=torch.bool).unsqueeze(0).to(self.device)

        with torch.no_grad():
            wav2vec_hidden = self.wav2vec_model(
                audio_, attention_mask=in_mask
            ).last_hidden_state
            # some times the output of wav2vec is 1 frame larger/smaller
            # correct for these cases
            size1 = wav2vec_hidden.shape[1]
            size2 = out_mask.shape[1]
            if size1 != size2:
                if size1 < size2:
                    out_mask = out_mask[:, :-1]
                else:
                    wav2vec_hidden = wav2vec_hidden[:, :-1, :]
            logits = self.sfc_model(wav2vec_hidden, out_mask)
            probs = torch.sigmoid(logits)
            probs[~out_mask] = 0
        probs = probs.detach().cpu().numpy()[0]

        segments = []
        sgm = Segment(0, len(probs), probs)
        sgm = trim(sgm, self.seg_thred)
        segments.append(sgm)
        
        return segments

    def policy(self, states: AgentStates) -> Union[TranslateAction, ReadAction]:
        assert states is not None

        if not hasattr(states, "last_segment_position"):
            states.last_segment_position = 0
        if not hasattr(states, "audio_start"):
            states.audio_start = None
        segment_finished = False

        segment_start = states.last_segment_position
        segment_end = len(states.source)

        if segment_end - segment_start >= self.max_segment_length * states.source_sample_rate:
            segment_finished = True
        if states.source_finished:
            segment_finished = True
        audio_ = torch.tensor([states.source[states.last_segment_position:]]).to(self.device)
        segments = self.shas_segment(audio_)
        offset = segments[0].offset * states.source_sample_rate + states.last_segment_position
        duration = segments[0].duration * states.source_sample_rate
        audio_end = math.ceil(offset + duration)
        if len(segments[0].probs)>0 and states.audio_start is None:
            states.audio_start = int(offset)  # Set the segment start point the first SHAS start
        
        if states.audio_start is None:
            return ReadAction()
        else:
            curr_segment_len = (segment_end - states.audio_start) / states.source_sample_rate
            if audio_end + self.end_bias < segment_end and curr_segment_len > self.min_segment_length:
                segment_finished = True

            if states.source_finished:
                states.last_segment_position = 0
            elif segment_finished:
                states.last_segment_position = segment_end

            # For saving the segmentation yaml file.
            if segment_finished and self.out_segment:
                states.segment_infos.append(
                    {
                        "duration": curr_segment_len,
                        "offset": states.audio_start / states.source_sample_rate,
                        "rW": 0,
                        "uW": 0,
                        "speaker_id": "NA",
                    }
                    )
            
            if states.audio_start is not None:
                logging.info(
                    f"Current from {states.audio_start / states.source_sample_rate} to {segment_end / states.source_sample_rate} seconds; source length: {len(states.source) / states.source_sample_rate}; segment finished: {segment_finished}"
                )
            return TranslateAction(
                states=states,
                speech_to_translate_start=states.audio_start,
                speech_to_translate_end=segment_end,
                segment_finished=segment_finished,
                source_finished=states.source_finished,
            )