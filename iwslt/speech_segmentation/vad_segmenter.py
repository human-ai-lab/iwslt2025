import logging
import torch
from typing import Union
from simuleval.agents.actions import ReadAction
from simuleval.agents.states import AgentStates
from ..speech_segmentation.speech_segmenter import SpeechSegmenter, TranslateAction
from silero_vad import get_speech_timestamps

class VADSegmenter(SpeechSegmenter):
    '''
    Segment audio based on Silero VAD
    https://github.com/snakers4/silero-vad?tab=readme-ov-file
    '''

    def __init__(self, args):
        self.max_segment_length = args.max_segment_length
        self.min_segment_length = args.min_segment_length
        self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.vad_thred = getattr(args, "vad_thred", None)
        self.max_pause_length = getattr(args, "max_pause_length", None)
        assert self.max_segment_length is not None, "Please set max_segment_length"
        assert self.vad_thred is not None, "Please set vad_thred"
        self.get_speech_timestamps, _, _, _, _= self.vad_utils

    @staticmethod
    def add_args(parser):
        parser.add_argument("--max-segment-length", default=20.0,
                            type=float, help="Maximum segment length in seconds")
        parser.add_argument("--min-segment-length", default=2.0,
                            type=float, help="Minimum segment length in seconds")
        parser.add_argument("--vad-thred", default=0.5,
                            type=float, help="Threshold for VAD")
        parser.add_argument("--max-pause-length", default=0.5,
                            type=float, help="Maximum pause length in a segment")

    def policy(self, states: AgentStates) -> Union[TranslateAction, ReadAction]:
        assert states is not None

        if not hasattr(states, "last_segment_position"):
            states.last_segment_position = 0
        
        max_pause_len = states.source_sample_rate * self.max_pause_length
        speech_timestamps = get_speech_timestamps(
            states.source[states.last_segment_position:],
            self.vad_model,
            threshold=self.vad_thred,
        )
        breakpoint()
        segment_start = states.last_segment_position
        segment_end = len(states.source)

        segment_finished = False
        if segment_end - segment_start >= self.max_segment_length * states.source_sample_rate:
            segment_finished = True
        if states.source_finished:
            segment_finished = True
        if len(speech_timestamps) > 0:
            vad_end = speech_timestamps[-1]['end'] + states.last_segment_position
            if (segment_end - vad_end > max_pause_len 
                and segment_end-segment_start > self.min_segment_length * states.source_sample_rate
                ):
                segment_finished = True

        if states.source_finished:
            states.last_segment_position = 0
        elif segment_finished:
            states.last_segment_position = segment_end

        logging.info(
            f"Current from {segment_start / states.source_sample_rate} to {segment_end / states.source_sample_rate} seconds; source length: {len(states.source) / states.source_sample_rate}; segment finished: {segment_finished}"
        )
        return TranslateAction(
            states=states,
            speech_to_translate_start=segment_start,
            speech_to_translate_end=segment_end,
            segment_finished=segment_finished,
            source_finished=states.source_finished,
        )