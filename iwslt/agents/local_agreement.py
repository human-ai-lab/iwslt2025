import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from argparse import ArgumentParser
from simul_agents import BaseAgent
from simuleval.agents.states import AgentStates
from simuleval.agents.actions import Action, WriteAction, ReadAction
from simuleval.agents import AgentPipeline
from simuleval.utils import entrypoint
from typing import Optional
import torch.nn.functional as F
import torch

class LocalAgreement(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.offline = getattr(args, "offline", False)
        if self.offline:
            self.args.out_attn = False

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--offline', action='store_true', help='offline mode')
        parser.add_argument('--data-root', type=str, help='data root')
        parser.add_argument('--target-lang', type=str, help='target language')
        parser.add_argument('--device', type=str, default='0', help='device to run the model')
        parser.add_argument('--model-path', type=str, help='path to the model checkpoint')
        parser.add_argument('--no-repeat-ngram-size', type=int, default=0, help='n-gram size for no repeat')
        parser.add_argument('--decode-config', type=str, default='iwslt/config/decode_config.yaml', help='Path to the decoding configuration')
        parser.add_argument('--decode-method', type=str, default='beam', help='Decoding method')
        parser.add_argument('--llm-name', type=str, default='llama', help='LLM name (e.g. llama, qwen, etc)')
        
    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states
        self._ensure_state_attributes(states)

        if states.prev_tokens is None:
            states.prev_tokens = torch.tensor([self.model.llama_tokenizer("Translation: ").input_ids], device=self.device)

        hypo = self.generate_hypothesis(states, self.args)
        
        if states.source_finished:
            prediction = self.decode_fn(hypo)[0].strip()
            return WriteAction(prediction, finished=True)
        else:
            stable_len = 0
            for stable_len, (_a, _b) in enumerate(zip(states.hypothesis, hypo[0])):
                if _a != _b:
                    break
                if stable_len == len(states.hypothesis) - 1:
                    stable_len += 1

            states.hypothesis = hypo[0][stable_len:]  # Discarded part
            new_stable_hypothesis = hypo[:, :stable_len]
            
            if new_stable_hypothesis.size(1) > 0:
                states.prev_tokens = torch.cat([states.prev_tokens, new_stable_hypothesis], dim=-1)
                prediction = self.decode_fn(new_stable_hypothesis)[0].strip()
                return WriteAction(prediction, finished=states.source_finished)
            else:
                return ReadAction()
            


@entrypoint
class S2TWithLAAgentPipeline(AgentPipeline):
    pipeline = [
        LocalAgreement,
    ]

    def __init__(self, args) -> None:
        s2t = self.pipeline[0](args)

        module_list = [s2t]

        super().__init__(module_list)

    @classmethod
    def from_args(cls, args):
        return cls(args)