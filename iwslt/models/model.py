from fairseq.models import register_model, register_model_architecture, BaseFairseqModel
from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform
import logging
import math 
from types import SimpleNamespace
from einops import rearrange
import json
import yaml
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer, StoppingCriteriaList, AutoModelForCausalLM
from .llama import LlamaForCausalLM
from .llama_tokenizer import ChatFormat
from .whisper import WhisperModel
from .Qformer import BertLMHeadModel, BertConfig
from peft import LoraConfig, TaskType, get_peft_model
from .util import StoppingCriteriaSub

logger = logging.getLogger(__name__)
WhisperProcessor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

@register_model("llm_sst")
class LlmSST(BaseFairseqModel):
    def __init__(self, encoder, projector, llama, llama_tokenizer, args):
        super().__init__()
        self.num_updates = 0
        self.args = args
        if getattr(args, 'spec_augment', False):
            logger.info("Applying SpecAugment to the input features")
            self.spec_augment = SpecAugmentTransform(
                time_warp_w=3,
                freq_mask_n=2,
                freq_mask_f=30,
                time_mask_n=3,
                time_mask_t=40,
                time_mask_p=0.1,
            )
        # Model components
        self.whisper_encoder = encoder
        if args.projector == 'pool':
            self.avgpool = projector
            output_dim = self.whisper_encoder.config.d_model

        elif args.projector == 'qformer':
            self.qformer, self.qformer_query_tokens = projector
            output_dim = self.qformer.config.hidden_size
        
        self.llama = llama
        self.llama_tokenizer = llama_tokenizer
        self.ln_speech = nn.LayerNorm(self.whisper_encoder.config.d_model)

        # Build projection mlp
        llm_hidden_size = self.llama.config.hidden_size
        linear_modules = [nn.Linear(output_dim, llm_hidden_size)]
        for _ in range(1, args.num_projection_layers):
            linear_modules.append(nn.GELU())
            linear_modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
        self.llama_proj = nn.Sequential(*linear_modules)
        
        # Prepare prompt for LLAMA
        with open(args.prompt_path, "r", encoding="utf-8") as f:
            raw_prompts = json.load(f)
        self.prompt = raw_prompts
        self.prompt_formater = ChatFormat(self.llama_tokenizer)

        # Generation
        decode_config = getattr(args, 'decode_config', 'iwslt/config/decode_config.yaml')
        decode_config = yaml.safe_load(open(decode_config))
        decode_method = getattr(args, 'decode_method', 'beam')
        logger.info(f"Decoding method: {decode_method}")
        if decode_method == 'beam':
            self.decode_config = decode_config['beam_search']
        else:
            raise ValueError(f"Invalid decoding method: {decode_method}")
        self.decode_config = SimpleNamespace(**self.decode_config)
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--projector', choices=['pool', 'qformer'], help='Projector type')
        parser.add_argument('--offline', action='store_true', help='Use full-utterance rather than streaming chunks')
        parser.add_argument('--chunk-size', type=int, help='Chunk size in miniseconds')
        parser.add_argument('--whisper-path', type=str, help='Path to the Whisper model')
        parser.add_argument('--freeze-whisper', action='store_true', help='Freeze the Whisper encoder')
        parser.add_argument('--freeze-qformer', action='store_true', help='Freeze the Qformer')
        parser.add_argument('--llm-name', type=str, help='LLM name/kind')
        parser.add_argument('--llm-path', type=str, help='Path to the LLM model')
        parser.add_argument('--vocab-save-as', type=str, default="txt", help='Save vocab as (txt, json)')
        parser.add_argument('--query-dim', type=int, help='Output dimension of the projector')
        parser.add_argument('--num-projection-layers', type=int, help='Number of layers in the projection MLP')
        parser.add_argument('--lora-rank', type=int, help='Rank of the Lora matrix')
        parser.add_argument('--lora-alpha', type=int, help='Alpha of the Lora matrix')
        parser.add_argument('--lora-dropout', type=float, help='Dropout rate for Lora')
        parser.add_argument('--prompt-path', type=str, default='', help='Path to the prompt file')
        parser.add_argument('--spec-augment', action='store_true', help='Apply SpecAugment to the input features')
        
        # Generation
        parser.add_argument('--decode-config', type=str, default='iwslt/config/decode_config.yaml', help='Path to the decoding configuration')
        parser.add_argument('--decode-method', type=str, default='beam', help='Decoding method')
        parser.add_argument('--out-attn', action='store_true', help='Output attentions during generation')


    @property
    def device(self):
        return list(self.parameters())[0].device

    @classmethod
    def init_speech_Qformer(cls, speech_width, num_hidden_layers=2, args=None):
        '''
        num_query_token: number of query tokens of Qformer
        speech_width: Dimension of the speech states
        '''
        if args.chunk_size > 30000:
            raise ValueError("Chunk size should be less than 30s")
        if args.offline:
            args.chunk_size = 30000
        qformer_config = BertConfig.from_pretrained("bert-base-uncased")
        qformer_config.num_hidden_layers = num_hidden_layers
        qformer_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        qformer_config.add_cross_attention = True
        qformer_config.cross_attention_freq = 1
        qformer_config.query_length = args.query_dim
        Qformer = BertLMHeadModel(config=qformer_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, args.query_dim, qformer_config.hidden_size)
        )  # The first dim is prepared for different chunks
        query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)

        return Qformer, query_tokens

    @classmethod
    def build_whisper(cls, args):
        ''' Build the Whisper-large-v3 encoder '''
        model_path = args.whisper_path
        encoder = WhisperModel.from_pretrained(model_path).encoder
        logger.info(f"loaded encoder from: {model_path}")
        if getattr(args, 'freeze_whisper', False):
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            encoder.eval()
            logging.info("Freezing Whisper encoder")

        return encoder

    @classmethod
    def build_qformer(cls, encoder_states_dim, args):
        ''' Build the Qformer decoder '''
        decoder, qformer_query_tokens = cls.init_speech_Qformer(encoder_states_dim, args=args)
        decoder.bert.embeddings.word_embeddings = None
        decoder.bert.embeddings.position_embeddings = None
        for layer in decoder.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        decoder.cls = None
        if getattr(args, 'freeze_qformer', False):
            for name, param in decoder.named_parameters():
                param.requires_grad = False
            decoder.eval()
            qformer_query_tokens.requires_grad = False
            logging.info("Freezing Qformer")

        return decoder, qformer_query_tokens

    @classmethod
    def build_pooling(cls, args):
        hw = int(args.query_dim ** 0.5)
        projector = nn.AdaptiveAvgPool2d((hw, hw))
        
        return projector

    @classmethod
    def build_llm(cls, args):
        """ Tokenizer """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism
        llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)

        llm_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        llm_tokenizer.add_special_tokens({'unk_token': '<unk>'})
        llm_tokenizer.add_tokens('<Speech>')
        llm_tokenizer.add_tokens('</Speech>')
        # llm_tokenizer.add_tokens('Transcription')
        llm_tokenizer.padding_side = 'right'

        # save vocab
        llm_name = args.llm_path.split("/")[-1]
        vocab = llm_tokenizer.get_vocab()

        if getattr(args,'vocab_save_as', 'txt') == "txt":
            vocab_path = os.path.join(args.user_dir, 'tokenizer/llm_tokenizer/', args.llm_name, f'{llm_name}_vocab.txt')
            if not os.path.exists(vocab_path):
                with open(vocab_path, "w", encoding="utf-8") as f:
                    for token, token_id in tqdm(sorted(vocab.items(), key=lambda x: x[1])):  # Sort by ID
                        f.write(f"{token} {token_id}\n")
        elif getattr(args,'vocab_save_as', 'txt') == "json":
            vocab_path = os.path.join(args.user_dir, 'tokenizer/llm_tokenizer/', args.llm_name, f'{llm_name}_vocab.json')
            if not os.path.exists(vocab_path):
                with open(vocab_path, 'w', encoding="utf-8") as f:
                    json.dump(dict(sorted(vocab.items(), key=lambda x: x[1])), f, indent=4, ensure_ascii=False)

                    
        """ Model """
        if (args.llm_name == "llama"):
            llm_model = LlamaForCausalLM.from_pretrained(args.llm_path)
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(args.llm_path)

        llm_model.resize_token_embeddings(len(llm_tokenizer))
        for name, param in llm_model.named_parameters():
            param.requires_grad = False
        logging.info(f"{llm_name} model loaded from: {args.llm_path}")
        
        # Lora
        lora_rank = args.lora_rank
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        llm_model = get_peft_model(llm_model, lora_config)
        llm_model.print_trainable_parameters()
        logging.info('LoRA Training.')

        return llm_model, llm_tokenizer

    
    @classmethod
    def build_model(cls, args, task=None):
        project_type = args.projector
        whisper = cls.build_whisper(args)

        if project_type == 'pool':
            projector = cls.build_pooling(args)

        elif project_type == 'qformer':
            projector = cls.build_qformer(whisper.config.d_model, args)

        else:
            raise ValueError(f"Invalid projector type: {projector}")

        llm, llm_tokenizer = cls.build_llm(args)

        return cls(whisper, projector, llm, llm_tokenizer, args)
    
    def extract_features(self, waveform, sample_rate=16000):
        '''
        Extract the Whisper features from the waveform

        Sampling rate: 16kHz
        hop_length: 160
        num_frames_per_second = 16000 / 160 = 100
        1 spectrogram frame = 10ms

        Args:
            waveform: List[np.array], each array is a waveform
        Return:
            features: B, 128, 3000
            attention_mask: B, 3000
        
        '''
        spec_aug = getattr(self.args, 'spec_augment', False) and self.training
        
        if spec_aug:
            return_tensors = None
        else:
            return_tensors = 'pt'

        processor_out = WhisperProcessor.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_attention_mask=True,
            do_normalize=True,
            device='cpu',
            return_tensors=return_tensors,
        )
        features = processor_out['input_features']  # numpy
        attention_mask = processor_out['attention_mask']

        if spec_aug:
            augmented = []
            for feat_ in features:
                feat_aug = self.spec_augment(feat_.T)
                feat_aug_tensor = torch.from_numpy(feat_aug.T)
                augmented.append(feat_aug_tensor)
            features = torch.stack(augmented).to(self.device)
        else:
            features = features.to(self.device)
        
        return features, attention_mask

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def gen_attn_mask(self, input_features, chunk_size, padding_mask=None):
        ''' Causal attention mask for the Whisper encoder '''
        bsz, _, seq_len = input_features.size()
        conv_out_length = math.ceil(seq_len/2)  # Whisper conv layer downsamples the input by 2
        num_frames_per_chunk = chunk_size // 20  # 1 frame of the whisper conv output represents 20ms
        chunk_idx = torch.arange(conv_out_length).to(input_features.device) // num_frames_per_chunk
        attn_mask = chunk_idx.unsqueeze(1) < chunk_idx.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0).expand(bsz, 1, -1, -1)  # bsz, head, seq_len, seq_len
        # Utilize the padding mask
        if padding_mask is not None:
            padding_mask = torch.tensor(padding_mask).to(input_features.device)
            # Downsample the padding mask to match the conv_out_length
            downsampled_padding_mask = padding_mask[:, ::2]
            padding_mask_expanded = downsampled_padding_mask.unsqueeze(1).unsqueeze(1)  # bsz, 1, 1, seq_len
            attn_mask = attn_mask | (padding_mask_expanded == 0)

        attn_mask_float = input_features.new_zeros(attn_mask.shape)
        attn_mask_float = attn_mask_float.masked_fill(attn_mask, -1e8 if input_features.dtype==torch.float32 else -1e4)

        return attn_mask_float, chunk_idx, downsampled_padding_mask

    def forward_whisper(self, waveform):
        '''
        Args:
            waveform: List[np.array], each array is a waveform
        Return:
            encoder_states: B, T, C
        '''
        if isinstance(waveform, torch.Tensor):
            if waveform.device.type == 'cuda':
                waveform = waveform.cpu()
            waveform = waveform.numpy()
        mel_features, pad_mask = self.extract_features(waveform)  # (B, 128, 3000) and (B, 3000)
        if getattr(self.args, 'fp16', False):
            mel_features = mel_features.half()

        if not self.args.offline:
            whisper_attn_mask, chunk_idx, source_padding_mask = self.gen_attn_mask(mel_features, self.args.chunk_size, pad_mask)
        else:
            whisper_attn_mask, chunk_idx, source_padding_mask = None, None, None

        encoder_states = self.whisper_encoder(input_features=mel_features, 
                                   attention_mask=whisper_attn_mask,
                                   output_attentions=None,
                                   output_hidden_states=None,
                                   return_dict=True).last_hidden_state

        return encoder_states, chunk_idx, source_padding_mask

    def forward_qformer(self, speech_states, chunk_idx):
        bsz = speech_states.size(0)
        if not self.args.offline:
            """ Seperate into chunks """
            num_chunks = chunk_idx.max().item() + 1
            chunks = []
            max_len = 0
            for i in range(num_chunks):
                selection = chunk_idx == i
                chunk = speech_states[:, selection, :]
                if chunk.size(1) > max_len:
                    max_len = chunk.size(1)
                if chunk.size(1) < max_len:
                    chunk = F.pad(chunk, (0, 0, 0, max_len - chunk.size(1)))
                chunks.append(chunk)

            chunks = torch.stack(chunks, dim=1)  # B, num_chunks, max_chunk_size, C
            chunks = chunks.view(-1, max_len, chunks.size(-1))  # B*num_chunks, T, C
        else:
            chunks = speech_states

        # Prepare the query tokens for each chunk in each mini-batch
        query_tokens = self.qformer_query_tokens.expand(chunks.size(0), -1, -1)
        # Q-former forward pass
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=chunks,
            encoder_attention_mask=None,
            return_dict=True
        ).last_hidden_state
        # Restore the chunked output to the original
        query_output = query_output.view(bsz, -1 , query_output.size(-1)).contiguous() # B, N*num_chunks, 768
        speech_atts = torch.ones(query_output.size()[:-1], dtype=torch.long).to(query_output.device)

        return query_output, speech_atts

    def forward_pooling(self, speech_states):
        speech_states = F.pad(speech_states, (0, 0, 0, 1521 - speech_states.size(1)))  # pad to 39*39
        hw = int(speech_states.size(1) ** 0.5)
        speech_states = rearrange(speech_states, "b (h w) c -> b c h w", h=hw, w=hw)  # B, C, H, W; T=H*W
        speech_states = self.avgpool(speech_states)
        speech_states = rearrange(speech_states, "b c h w -> b (h w) c")  # B, T, C

        return speech_states

    def wrap_prompt(self, speech_embeds, formated_prompts, atts):

        def prompt_process(speech_embed, formated_prompt):
            first_speech_start_tag_id = formated_prompt.index(self.llama_tokenizer.convert_tokens_to_ids('<Speech>'))
            first_speech_end_tag_id = formated_prompt.index(self.llama_tokenizer.convert_tokens_to_ids('</Speech>'))
            speech_start_idx = formated_prompt.index(self.llama_tokenizer.convert_tokens_to_ids('<Speech>'), first_speech_start_tag_id+1)
            speech_end_idx = formated_prompt.index(self.llama_tokenizer.convert_tokens_to_ids('</Speech>'), first_speech_end_tag_id+1)
            # speech_start_idx = formated_prompt.index(self.llama_tokenizer.convert_tokens_to_ids('<Speech>'))
            # speech_end_idx = formated_prompt.index(self.llama_tokenizer.convert_tokens_to_ids('</Speech>'))
            p_before_tokens = torch.tensor(formated_prompt[:speech_start_idx+1]).unsqueeze(0).to(speech_embeds.device)
            p_after_tokens = torch.tensor(formated_prompt[speech_end_idx:]).unsqueeze(0).to(speech_embeds.device)
            p_before_embeds = self.llama.model.model.embed_tokens(p_before_tokens)
            p_after_embeds = self.llama.model.model.embed_tokens(p_after_tokens)
            wrapped_embeds = torch.cat([p_before_embeds, speech_embed, p_after_embeds], dim=1)
            # Get speech embeds start/end positions
            speech_start = p_before_tokens.size(1)
            speech_end = speech_start + speech_embed.size(1)
            p_before_len = p_before_tokens.size(1)
            p_after_len = p_after_tokens.size(1)

            return wrapped_embeds, speech_start, speech_end, p_before_len, p_after_len
        
        if len(formated_prompts) > 1:
            # Padding
            max_len = max([len(p) for p in formated_prompts])
            for i, p in enumerate(formated_prompts):
                if len(p) < max_len:
                    formated_prompts[i] = p + [self.llama_tokenizer.pad_token_id] * (max_len - len(p))
        all_wrapped_embeds = []
        speech_positions = []
        wrapped_atts = []
        for i, formated_prompt in enumerate(formated_prompts):
            wrapped_embeds, speech_start, speech_end, p_before_len, p_after_len = prompt_process(speech_embeds[i].unsqueeze(0), formated_prompt)
            all_wrapped_embeds.append(wrapped_embeds)
            speech_positions.append((speech_start, speech_end))
            p_before_att = torch.ones([1, p_before_len], dtype=torch.long).to(speech_embeds.device)
            p_after_att = torch.ones([1, p_after_len], dtype=torch.long).to(speech_embeds.device)
            wrapped_att = torch.cat([p_before_att, atts[i].unsqueeze(0), p_after_att], dim=1)
            wrapped_atts.append(wrapped_att)
        all_wrapped_embeds = torch.cat(all_wrapped_embeds, dim=0)
        wrapped_atts = torch.cat(wrapped_atts, dim=0)
        
        return all_wrapped_embeds, wrapped_atts, speech_positions

    def forward(self, src_tokens, src_lengths, src_texts=None, targets=None, tgt_langs=None, prev_output_tokens=None):
        src_texts[src_texts==self.llama_tokenizer.eos_token_id]=self.llama_tokenizer.pad_token_id
        # targets[targets==self.llama_tokenizer.eos_token_id]=self.llama_tokenizer.pad_token_id
        speech_embeds, chunk_idx, source_padding_mask = self.forward_whisper(src_tokens)
        speech_embeds = self.ln_speech(speech_embeds)  # layernorm to the encoder output
        bsz = speech_embeds.size(0)
        if self.args.projector == 'pool':
            speech_embeds = self.forward_pooling(speech_embeds)
        elif self.args.projector == 'qformer':
            speech_embeds, _ = self.forward_qformer(speech_embeds, chunk_idx)  # B, N*num_chunks, 768
        speech_embeds = self.llama_proj(speech_embeds)  # MLP projection

        if not self.args.offline:
            num_chunks = chunk_idx.max().item() + 1
            self.args.num_chunks = num_chunks
            chunk_length = speech_embeds.size(1) // num_chunks
            # Create padding_mask for Llama according to the source_padding_mask
            num_valid_source_tokens = (source_padding_mask == 1).sum(dim=-1)  # valid token number for each batch
            num_valid_chunks = torch.ceil(num_valid_source_tokens.float() / chunk_length).long()
            num_valid_speech_embeds = num_valid_chunks * self.args.num_query_token
            speech_padding_mask = speech_embeds.new_zeros(bsz, speech_embeds.size(1))
            for i in range(bsz):
                speech_padding_mask[i, :num_valid_speech_embeds[i]] = 1
        else:
            speech_padding_mask = torch.ones([bsz, speech_embeds.size(1)], dtype=torch.long).to(speech_embeds.device)
            
        # Wrap the speech_embeds with the prompt
        batch_prompts = [self.prompt['st'][0] for _ in range(bsz)]
        for i_, batch_prompt in enumerate(batch_prompts):
            if tgt_langs[i_] == 'de':
                lang_ = 'German'
            elif tgt_langs[i_] == 'zh':
                lang_ = 'Chinese'
            elif tgt_langs[i_] == 'ja':
                lang_ = 'Japanese'
            elif tgt_langs[i_] == 'en':
                lang_ = 'English'
            else:
                raise ValueError(f"Invalid target language: {tgt_langs[i_]}")
            batch_prompt[1]['content'] = batch_prompt[1]['content'].replace("<tgt_lang>", lang_)
        
        formated_prompts = [self.prompt_formater.encode_dialog_prompt(p) for p in batch_prompts]
        speech_embeds, speech_padding_mask, speech_positions = self.wrap_prompt(speech_embeds, formated_prompts, speech_padding_mask)
        
        """ Prepare text input and target for LLAMA """
        eos = torch.ones(
            [bsz, 1],
            dtype=torch.long,
            device=speech_embeds.device,
        )*self.llama_tokenizer.eos_token_id
        # pre_targets = self.llama_tokenizer.encode("Translation: ", add_special_tokens=False)
        # pre_targets = torch.tensor(pre_targets).to(speech_embeds.device).unsqueeze(0)
        # pre_targets = pre_targets.expand(bsz, -1)
        # targets[(targets==self.llama_tokenizer.pad_token_id).cumsum(dim=-1)==1]=self.llama_tokenizer.eos_token_id
        # targets = torch.cat([pre_targets, targets], dim=1)
        # pre_src_texts = self.llama_tokenizer.encode(" Transcription: ", add_special_tokens=False)
        # pre_src_texts = torch.tensor(pre_src_texts).to(speech_embeds.device).unsqueeze(0)
        # pre_src_texts = pre_src_texts.expand(bsz, -1)
        # src_texts = torch.cat([pre_src_texts, src_texts], dim=1)

        # to_regress_tokens = torch.cat([targets, src_texts], dim=1)

        """ pytorch loss function treats -100 as a special ignore index """
        """ Note: The Llama loss function do shifting for targets. There's no need to shift the targets here. """
        to_regress_tokens = torch.cat([targets, eos], dim=1)
        targets = to_regress_tokens.masked_fill(to_regress_tokens == self.llama_tokenizer.pad_token_id, -100) # ignore padded positions in the target sequence
        
        empty_targets = (
            torch.ones(
                [bsz, speech_embeds.size(1)],
                dtype=torch.long,
            ).to(speech_embeds.device).fill_(-100)
        )  # The targets of the speech_embeds should be ignored
        combined_targets = torch.cat([empty_targets, targets], dim=1)  # speech_embeds+text+eos

        # Prepare the inputs for LLAMA
        input_tokens = to_regress_tokens
        text_embeds = self.llama.model.model.embed_tokens(input_tokens)  # embeddings
        input_embeds = torch.cat([speech_embeds, text_embeds], dim=1)

        # Concat input padding masks
        text_atts = torch.ones([bsz, text_embeds.size(1)], dtype=torch.long).to(text_embeds.device)
        input_padding_mask = torch.cat([speech_padding_mask, text_atts], dim=1)
        text_start = speech_embeds.size(1) + 1 
        text_stop = input_embeds.size(1)
        text_positions = (text_start, text_stop)
        
        # calulate loss
        outputs = self.llama(
            inputs_embeds=input_embeds,
            attention_mask=input_padding_mask,
            return_dict=True,
            labels=combined_targets,
            speech_embeds_positions=speech_positions,
            text_embeds_positions=text_positions,
            args=self.args,
        )
        loss = outputs.loss

        return {"loss": loss}

    def decode_fn(self, tokens):
        return self.llama_tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def llama_generate(self, speech_embeds, speech_padding_mask, decode_config=None, args=None):
        stop_words_ids = [torch.tensor([self.llama_tokenizer.eos_token_id]).to(speech_embeds.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        out_attn = getattr(args, 'out_attn', False)
        outputs = self.llama.generate(
            inputs_embeds=speech_embeds,
            attention_mask=speech_padding_mask,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
            max_new_tokens=getattr(decode_config, 'max_new_tokens', 100),
            num_beams=getattr(decode_config, 'beam_num', 1),
            do_sample=getattr(decode_config, 'do_sample', False),
            min_length=getattr(decode_config, 'min_length', 1),
            top_p=getattr(decode_config, 'top_p', None),
            length_penalty=getattr(decode_config, 'length_penalty', 0.0),
            repetition_penalty=getattr(decode_config, 'repetition_penalty', 1.0),
            temperature=getattr(decode_config, 'temperature', 1.0),
            output_attentions=out_attn,
            return_dict_in_generate=True,
        )

        return outputs

    def generate(self, src_tokens, prev_tokens=None, tgt_langs=None, args=None):
        speech_embeds, chunk_idx, source_padding_mask = self.forward_whisper(src_tokens)
        speech_embeds = self.ln_speech(speech_embeds)  # layernorm to the encoder output
        bsz = speech_embeds.size(0)
        if self.args.projector == 'pool':
            speech_embeds = self.forward_pooling(speech_embeds)
        elif self.args.projector == 'qformer':
            speech_embeds, _ = self.forward_qformer(speech_embeds, chunk_idx)
        speech_embeds = self.llama_proj(speech_embeds)
        speech_padding_mask = torch.ones([bsz, speech_embeds.size(1)], dtype=torch.long).to(speech_embeds.device)

        batch_prompts = [self.prompt['st'][0] for _ in range(bsz)]
        target_lang = getattr(args, 'target_lang', None)
        if target_lang is not None:
            # Inference
            if target_lang == 'de':
                target_lang = 'German'
            elif target_lang == 'zh':
                target_lang = 'Chinese'
            elif target_lang == 'ja':
                target_lang = 'Japanese'
            elif target_lang == 'en':
                target_lang = 'English'
            else:
                raise ValueError(f"Invalid target language: {target_lang}")
            for i_, batch_prompt in enumerate(batch_prompts):
                batch_prompt[1]['content'] = batch_prompt[1]['content'].replace("<tgt_lang>", target_lang)
        else:
            # Training
            assert tgt_langs is not None
            for i_, batch_prompt in enumerate(batch_prompts):
                if tgt_langs[i_] == 'de':
                    lang_ = 'German'
                elif tgt_langs[i_] == 'zh':
                    lang_ = 'Chinese'
                elif tgt_langs[i_] == 'ja':
                    lang_ = 'Japanese'
                elif tgt_langs[i_] == 'en':
                    lang_ = 'English'
                else:
                    raise ValueError(f"Invalid target language: {tgt_langs[i_]}")
                batch_prompt[1]['content'] = batch_prompt[1]['content'].replace("<tgt_lang>", lang_)

        formated_prompts = [self.prompt_formater.encode_dialog_prompt(p) for p in batch_prompts]
        speech_embeds, speech_padding_mask, speech_positions = self.wrap_prompt(speech_embeds, formated_prompts, speech_padding_mask)

        if prev_tokens is not None:
            prev_embeds = self.llama.model.model.embed_tokens(prev_tokens)
            input_embeds = torch.cat([speech_embeds, prev_embeds], dim=1)
            speech_padding_mask = torch.cat([speech_padding_mask, torch.ones([bsz, prev_embeds.size(1)], dtype=torch.long).to(speech_embeds.device)], dim=1)
        else:
            input_embeds = speech_embeds
        outputs = self.llama_generate(input_embeds, speech_padding_mask, self.decode_config, args)
        hypos = outputs['sequences']

        if (args.llm_name == "qwen"):
            return hypos

        else:
            if getattr(args, 'out_attn', False):
                attns = outputs['attentions']  # new_tokens, layers, tensor(beams, num_heads, seq_len, seq_len)
                return hypos, attns, speech_positions
            else:
                return hypos


@register_model_architecture(model_name="llm_sst", arch_name="llm_sst_pool_8B")
def avg_pool_architecture(args):
    args.whisper_path = getattr(args, 'whisper_path', 'openai/whisper-large-v3')  # Path to Whisper model
    args.llm_name = getattr(args, 'llm_name', "llama")  # LLM name (e.g. llama, qwen, etc)
    args.llm_path = getattr(args, 'llm_path', "meta-llama/Meta-Llama-3.1-8B-Instruct")  # Path to LLM model
    args.freeze_whisper = getattr(args, 'freeze_whisper', True)  # Whether to freeze Whisper encoder
    args.num_projection_layers = getattr(args, 'num_projection_layers', 2)
    args.projector = getattr(args, 'projector', 'pool')  # Projector type
    args.query_dim = getattr(args, 'query_dim', 256)
    args.lora_rank = getattr(args, 'lora_rank', 8)  # LoRA matrix rank
    args.lora_alpha = getattr(args, 'lora_alpha', 32)  # LoRA scaling alpha
    args.lora_dropout = getattr(args, 'lora_dropout', 0.1)  # LoRA dropout rate
    args.chunk_size = getattr(args, 'chunk_size', 1000)
    logger.info("AdaptiveAvgPool2D architecture has been registered!")

@register_model_architecture(model_name="llm_sst", arch_name="llm_sst_pool")
def avg_pool_architecture(args):
    args.whisper_path = getattr(args, 'whisper_path', 'openai/whisper-large-v3')  # Path to Whisper model
    args.llm_name = getattr(args, 'llm_name', "llama")  # LLM name (e.g. llama, qwen, etc)
    args.llm_path = getattr(args, 'llm_path', "meta-llama/Llama-3.2-3B-Instruct")  # Path to LLM model
    args.freeze_whisper = getattr(args, 'freeze_whisper', True)  # Whether to freeze Whisper encoder
    args.num_projection_layers = getattr(args, 'num_projection_layers', 2)
    args.projector = getattr(args, 'projector', 'pool')  # Projector type
    args.query_dim = getattr(args, 'query_dim', 256)
    args.lora_rank = getattr(args, 'lora_rank', 8)  # LoRA matrix rank
    args.lora_alpha = getattr(args, 'lora_alpha', 32)  # LoRA scaling alpha
    args.lora_dropout = getattr(args, 'lora_dropout', 0.1)  # LoRA dropout rate
    args.chunk_size = getattr(args, 'chunk_size', 1000)
    logger.info("AdaptiveAvgPool2D architecture has been registered!")

@register_model_architecture(model_name="llm_sst", arch_name="llm_sst_qformer_base")
def avg_pool_architecture(args):
    args.whisper_path = getattr(args, 'whisper_path', 'openai/whisper-large-v3')  # Path to Whisper model
    args.llm_name = getattr(args, 'llm_name', "llama")  # LLM name (e.g. llama, qwen, etc)
    args.llm_path = getattr(args, 'llm_path', 'meta-llama/Llama-3.1-8B')  # Path to LLM model
    args.freeze_whisper = getattr(args, 'freeze_whisper', True)  # Whether to freeze Whisper encoder
    args.num_projection_layers = getattr(args, 'num_projection_layers', 1)
    args.projector = getattr(args, 'projector', 'qformer')  # Projector type
    args.query_dim = getattr(args, 'query_dim', 256)
    args.lora_rank = getattr(args, 'lora_rank', 8)  # LoRA matrix rank
    args.lora_alpha = getattr(args, 'lora_alpha', 32)  # LoRA scaling alpha
    args.lora_dropout = getattr(args, 'lora_dropout', 0.1)  # LoRA dropout rate
    args.chunk_size = getattr(args, 'chunk_size', 1000)
    logger.info("Qformer Base Architecture has been registered!")


