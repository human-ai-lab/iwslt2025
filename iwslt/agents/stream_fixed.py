from iwslt.speech_segmentation.fixed_length_segmenter import FixedLengthSegmenter
from simuleval.agents import AgentPipeline
from simuleval.utils import entrypoint
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from argparse import ArgumentParser
from simul_agents import BaseAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import Action, WriteAction, ReadAction
from iwslt.speech_segmentation.speech_segmenter import SpeechSegmenter, TranslateAction
from typing import Optional
import torch.nn.functional as F
import torch

class StreamState(AgentStates):
    def reset(self):
        super().reset()

class StreamAgent(BaseAgent):
    def __init__(self, args, segmenter: SpeechSegmenter):
        super().__init__(args)
        self.segmenter = segmenter
        self.segment_counter = 0
        self.step_length = args.step_length
        self.hypo_policy = getattr(args, "hypo_policy", "local_agreement")

        self.offline = getattr(args, "offline", False)
        if self.offline or self.hypo_policy == "local_agreement":
            self.args.out_attn = False
        self.attn_layer = getattr(args, "attn_layer_num", -1)
        self.last_n_frame = getattr(args, "last_n_frame", None)
        self.attn_thred = getattr(args, "attn_thred", None)

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--offline', action='store_true', help='offline mode')
        parser.add_argument('--data-root', type=str, help='data root')
        parser.add_argument('--target-lang', type=str, help='target language')
        parser.add_argument('--device', type=str, default='0', help='device to run the model')
        parser.add_argument('--model-path', type=str, help='path to the model checkpoint')
        parser.add_argument('--decode-config', type=str, default='iwslt/config/decode_config.yaml', help='Path to the decoding configuration')
        parser.add_argument('--decode-method', type=str, default='beam', help='Decoding method')
        parser.add_argument('--out-attn', action='store_true', help='output attention weights')

        parser.add_argument('--hypo-policy', type=str, help='policy to use')
        parser.add_argument('--last-n-frame', type=int, default=2, help='last n frames')
        parser.add_argument('--attn-thred', type=float, help='threshold for EDAtt')
        parser.add_argument("--step-length", type=float, default=1.0,)
        parser.add_argument("--history-tokens", type=int, default=100, help="Maximum Number of tokens to keep in history")
        parser.add_argument('--llm-name', type=str, default='llama', help='LLM name (e.g. llama, qwen, etc)')

    def build_states(self) -> AgentStates:
        states = StreamState()

        return states

    def _local_agreement(self, states: AgentStates, hypo, semgment_or_source_finished):
        if semgment_or_source_finished:
            new_hypo = hypo
        else:
            stable_len = 0
            for stable_len, (_a, _b) in enumerate(zip(states.hypothesis, hypo[0])):
                if _a != _b:
                    break
                if stable_len == len(states.hypothesis) - 1:
                    stable_len += 1

            states.hypothesis = hypo[0][stable_len:]  # Discarded part
            hypo = hypo[:, :stable_len]
            if hypo.size(1) > 0:
                new_hypo = hypo
            else:
                new_hypo = None 

        return new_hypo

    def _edatt(self, states: AgentStates, hypo, attns, speech_positions, semgment_or_source_finished):
        if semgment_or_source_finished:
            new_hypo = hypo
        else:
            speech_positions = speech_positions[0]
            speech_start, speech_end = speech_positions
            end_idx = None
            for idx, token_attn in enumerate(attns):
                token_attn = torch.stack(token_attn)
                token_attn = torch.mean(token_attn, dim=0)
                token_attn = token_attn.cpu()[0].mean(dim=0)[0]
                audio_attn = token_attn[speech_start:speech_end]
                audio_attn = audio_attn.view(32, 8).mean(dim=1)
                audio_attn = F.softmax(audio_attn, dim=-1)
                attn_sum = audio_attn[-self.last_n_frame:].sum()
                if attn_sum > self.attn_thred:
                    end_idx = idx
                    break
            if end_idx is not None:
                new_hypo = hypo[:, :end_idx]
            else:
                new_hypo = hypo

        if new_hypo.size(1) > 0:
            states.prev_tokens = torch.cat([states.prev_tokens, new_hypo], dim=-1)
            states.prev_tokens = states.prev_tokens[:, -self.args.history_tokens:]
            if not torch.equal(states.prev_tokens[:, :3], self.prefix):
                states.prev_tokens = torch.cat([self.prefix, states.prev_tokens], dim=-1)
            translation = self.decode_fn(new_hypo)[0]
        else:
            translation = ""

        return translation

    def generate_hypothesis(self, _source, _prev_tokens, args):
        """
        This method takes *states* and *prefix_tokens* as inputs and generates and returns the mostly likely
        translation hypothesis *hypo*.
        """
        output = self.model.generate(_source, _prev_tokens, args=args)
        
        return output

    def _translate(self, states: AgentStates, segmenter_action: TranslateAction):
        semgment_or_source_finished = (
            segmenter_action.segment_finished
            or segmenter_action.source_finished
        )

        speech_to_translate = segmenter_action.speech_to_translate()
        if len(speech_to_translate) < 4000:
            return ""
        
        output = self.generate_hypothesis(speech_to_translate, states.prev_tokens, self.args)

        if self.offline or self.hypo_policy == "local_agreement":
            hypo = output
            attns = None
            speech_positions = None
            if states.prev_tokens is None:
                self.prefix = hypo[:, :3]
                states.prev_tokens = hypo[:, :3]  # "Translation: "
                hypo = hypo[:, 3:]  # Remove 'Translation: '
            if hypo.size(1) > 0:
                if hypo[:, -1] == self.eos_id:
                    hypo = hypo[:, :-1]  # remove eos token
            else:
                return ""
        else:
            hypo, attns, speech_positions = output

            if states.prev_tokens is None:
                self.prefix = hypo[:, :3]
                states.prev_tokens = hypo[:, :3]  # "Translation: "
                hypo = hypo[:, 3:]  # Remove 'Translation: '
                attns = attns[3:]

            if hypo[:, -1] == self.eos_id:
                hypo = hypo[:, :-1]  # remove eos token
                attns = attns[:-1]

        if self.hypo_policy == "edatt":
            pass
            # translation = self._edatt(states, hypo, attns, speech_positions, semgment_or_source_finished)
        elif self.hypo_policy == "local_agreement":
            new_hypo = self._local_agreement(states, hypo, semgment_or_source_finished)
            # new_hypo = hypo
        else:
            raise ValueError(f"Unknown policy: {self.hypo_policy}")
        
        if new_hypo is not None:
            states.prev_tokens = torch.cat([states.prev_tokens, new_hypo], dim=-1)
            states.prev_tokens = states.prev_tokens[:, -self.args.history_tokens:]
            if not torch.equal(states.prev_tokens[:, :3], self.prefix):
                states.prev_tokens = torch.cat([self.prefix, states.prev_tokens], dim=-1)
            translation = self.decode_fn(new_hypo)[0]
        else:
            translation = ""
        
        return translation
    
    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states
        self._ensure_state_attributes(states)
        segmenter_action = self.segmenter.policy(states)

        if isinstance(segmenter_action, TranslateAction):
            
            if (
                segmenter_action.segment_length() >= states.last_translated +
                (self.step_length if not segmenter_action.segment_finished else 0)
                or segmenter_action.source_finished
            ):
                new_translation = self._translate(states, segmenter_action)
                logging.info(f"translated: {new_translation}")
                states.last_translated = segmenter_action.segment_length()
                if segmenter_action.segment_finished:
                    states.prev_tokens = None
                    states.last_translated = 0
                    states.stable_hypothesis = []
                    states.hypothesis = []
                return WriteAction(
                    new_translation,
                    segmenter_action.source_finished,
                )
            else:
                return ReadAction()
        elif isinstance(segmenter_action, ReadAction):
            return segmenter_action  # Pass the read action
        else:
            raise ValueError(
                f"Unknown action type; got {type(segmenter_action)}, expected {TranslateAction} or {ReadAction}")

@entrypoint
class StreamSTAgentPipeline(AgentPipeline):
    pipeline = [
        StreamAgent,
    ]

    def __init__(self, args) -> None:
        s2t = self.pipeline[0](args, FixedLengthSegmenter(args))

        module_list = [s2t]

        super().__init__(module_list)

    @staticmethod
    def add_args(parser):
        StreamAgent.add_args(parser)
        FixedLengthSegmenter.add_args(parser)

    @classmethod
    def from_args(cls, args):
        return cls(args)