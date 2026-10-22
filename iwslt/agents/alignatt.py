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

class AlignAtt(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.offline = getattr(args, "offline", False)
        if self.offline or args.llm_name == 'qwen':
            self.args.out_attn = False
        self.attn_layer = getattr(args, "attn_layer_num", -1)
        self.last_n_frame = getattr(args, "last_n_frame", None)
        self.llm_name = getattr(args, "llm_name", "llama")

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
        parser.add_argument('--out-attn', action='store_true', help='output attention weights')

        parser.add_argument('--attn-layer-num', type=int, default=-1, help='layer number to extract attn')
        parser.add_argument('--last-n-frame', type=int, required=True, help='n_frame for AlignAtt')
        parser.add_argument('--llm-name', type=str, default='llama', help='LLM name (e.g. llama, qwen, etc)')  
        
    def policy(self, states: Optional[AgentStates] = None) -> Action:
        if states is None:
            states = self.states
        output = self.generate_hypothesis(states, self.args)

        if self.offline or self.llm_name == "qwen" :
            hypo = output
        else:
            hypo, attns, speech_positions = output

        if states.prev_tokens is None:
            states.prev_tokens = hypo[:, :3]  # "Translation: "
            hypo = hypo[:, 3:]  # Remove 'Translation: '
            attns = attns[3:]
        
        if hypo[:, -1] == self.eos_id:
            hypo = hypo[:, :-1]  # remove eos token
            attns = attns[:-1]

        if states.source_finished:
            prediction = self.decode_fn(hypo)[0]
            prediction = prediction.split('Transcription:')[0].strip()  # In case model out Transcription
            return WriteAction(prediction, finished=True)
        else:
            speech_positions = speech_positions[0]
            speech_start, speech_end = speech_positions
                
            end_idx = None
            # audio_attns = []
            for idx, token_attn in enumerate(attns):
                # token_attn = token_attn[self.attn_layer]
                token_attn = torch.stack(token_attn)
                token_attn = torch.mean(token_attn, dim=0)
                token_attn = token_attn.cpu()[0].mean(dim=0)[0]
                audio_attn = token_attn[speech_start:speech_end]
                audio_attn = audio_attn.view(32, 8).mean(dim=1)
            #     audio_attns.append(audio_attn)
            # audio_attns = torch.stack(audio_attns, dim=0)
            # normed_attns = F.softmax(audio_attns, dim=-1).numpy()
            # self.plot_attn(normed_attns, save_path=f'attn_layer_mean_2000.png')
            # pdb.set_trace()
                audio_attn = F.softmax(audio_attn, dim=-1)
                most_attended_frame = audio_attn.argmax()
                if len(audio_attn) - most_attended_frame <= self.last_n_frame:
                    end_idx = idx
                    break
            
            if end_idx is not None:
                new_hypo = hypo[:, :end_idx]
            else:
                new_hypo = hypo
                
            if new_hypo.size(1) > 0:
                states.prev_tokens = torch.cat([states.prev_tokens, new_hypo], dim=-1)
                prediction = self.decode_fn(new_hypo)[0]

                return WriteAction(prediction, finished=states.source_finished)
            else:
                return ReadAction()
            
            


@entrypoint
class S2TWithAlignAttAgentPipeline(AgentPipeline):
    pipeline = [
        AlignAtt,
    ]

    def __init__(self, args) -> None:
        s2t = self.pipeline[0](args)

        module_list = [s2t]
        super().__init__(module_list)

    @classmethod
    def from_args(cls, args):
        return cls(args)