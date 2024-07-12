# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import *

from ..videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM

class TimeTokenizer(nn.Module):
    def __init__(self, num_float_tokens, input_dim=4096, dropout_rate = 0.1):
        super(TimeTokenizer, self).__init__()
        # Learnable normalization parameters
#         self.scale = nn.Parameter(torch.tensor([[1.0]]))
#         self.shift = nn.Parameter(torch.tensor([[0.0]]))

        # Encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(1, num_float_tokens),
            nn.ReLU(),
            nn.Linear(num_float_tokens, int(num_float_tokens * 2)),
            nn.ReLU(),
            nn.Linear(int(num_float_tokens * 2), num_float_tokens)
        )
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Add dropout after the first ReLU activation
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Add dropout after the second ReLU activation
            nn.Linear(input_dim // 4, 1)
        )
        
    def encode(self, x):
        
        # Normalize input using affine transformation
#         x = x * self.scale + self.shift
        x = self.encoder(x)
        return F.softmax(x, dim=-1)

    def decode(self, x):
        x = self.decoder(x)
#         x = (x - self.shift) / self.scale
        # Denormalize output using the inverse of the affine transformation
        return x
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

class Videollama2MistralConfig(MistralConfig):
    model_type = "videollama2_mistral"

class Videollama2MistralModel(Videollama2MetaModel, MistralModel):
    config_class = Videollama2MistralConfig

    def __init__(self, config: MistralConfig):
        super(Videollama2MistralModel, self).__init__(config)

class Videollama2MistralForCausalLM(MistralForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2MistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)
        self.model = Videollama2MistralModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mm_use_time_token = config.mm_use_time_token
        
        ## load time mlp ##
        if self.mm_use_time_token:
            self.float_token_id_start = config.float_token_id_start
            self.float_token_id_end = config.float_token_id_end
            self.model.float_tokenizer = TimeTokenizer(self.float_token_id_end-self.float_token_id_start+1)
            self.range_tokens = config.range_tokens
            self.sorted_tokens = sorted(float(val) for val in self.range_tokens)
            
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        modal_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images_or_videos is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                X_modalities=[images_or_videos, modal_list]
            )
        else:
            inputs_embeds = self.get_embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

    def greedy_search(
        self,
        input_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.Tensor]:
        # init values
        input_ids = input_ids.to(torch.float32)
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = self.wrap_float_token(outputs.hidden_states[-1][:, -1, :].clone(), next_tokens)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        

    def wrap_float_token(self, last_hidden_states, next_tokens):
        next_tokens = next_tokens.to(torch.float32)
        if self.mm_use_time_token:
            # Determine which tokens are in the range for special handling
            float_mask = (self.float_token_id_start <= next_tokens) & (next_tokens <= self.float_token_id_end)

            # Apply the time MLP to the last hidden states corresponding to these tokens
            if float_mask.any():
                print(next_tokens)
                # Extract the last hidden states for tokens requiring fractional adjustment
                float_hidden_states = last_hidden_states[float_mask]

                # Calculate the fractional parts using time_mlp
                frac_part = self.model.float_tokenizer.decode(float_hidden_states)

                # Add the fractional part to the original next tokens in place
                next_tokens[float_mask] = self.float_to_token(frac_part)
        return next_tokens
    
    def get_embed_tokens(self, token_ids):
        # Convert token_ids to a torch tensor if it's not already
        if self.mm_use_time_token and len(token_ids):
#             if not isinstance(token_ids, torch.Tensor):
#                 token_ids = torch.tensor(token_ids, dtype=torch.long).to(self.model.device)

            # Identify the indices with floating token IDs
            is_float = (self.float_token_id_start <= token_ids) & (token_ids <= self.float_token_id_end)
            
            # Process floating token indices
            if is_float.any():
                # Allocate a tensor for embeddings that matches the input tokens
                embeddings = torch.zeros(*token_ids.shape, 4096, dtype = self.model.embed_tokens.weight.dtype, device=token_ids.device)
                
                # Handle non-floating tokens normally
                if (~is_float).any():
                    embeddings[~is_float] += self.model.embed_tokens(token_ids[~is_float].long())

                # Handle floating tokens
                float_indices = token_ids[is_float]
                float_input = self.token_to_float(float_indices).unsqueeze(-1).to(self.model.embed_tokens.weight.dtype)
                
                weights = self.model.float_tokenizer.encode(float_input)  # Get linear weights for each base embedding
                # Get the base embeddings
                embedding_float_base = self.model.embed_tokens(torch.arange(self.float_token_id_start, self.float_token_id_end + 1, device=self.model.device))
                
                # Compute weighted sum of base embeddings
                float_embedding = torch.matmul(weights, embedding_float_base)
                embeddings[is_float] += float_embedding
                return embeddings
            else:
                return self.model.embed_tokens(token_ids.long())
        else:
            return self.model.embed_tokens(token_ids.long())

    def token_to_float(self, token_ids):
        device = token_ids.device
        original_shape = token_ids.shape

        # Reshape token_ids to a one-dimensional array
        token_ids = token_ids.view(-1)

        # Calculate base indices for each token
        base_ids = token_ids.long() - self.float_token_id_start

        # Collect lower and upper bounds from pre-defined ranges
        lower_bounds = torch.tensor([float(self.range_tokens[idx]) for idx in base_ids], dtype = torch.float32, device=device)
        upper_bounds = torch.tensor([float(self.range_tokens[min(idx + 1, len(self.range_tokens) - 1)]) for idx in base_ids], dtype = torch.float32, device=device)

        # Calculate fractional parts for interpolation
        fractional_parts = token_ids.frac()

        # Perform interpolation
        interpolated_floats = lower_bounds + fractional_parts * (upper_bounds - lower_bounds)

        # Reshape interpolated_floats to match the original shape of token_ids
        interpolated_floats = interpolated_floats.view(original_shape)

        # Round the result to two decimal places
        return interpolated_floats

    
    def float_to_token(self, floats):
        # Clip the input values to be within the specified range
        floats_clipped = torch.clamp(floats, min=float(self.range_tokens[0]), max=float(self.range_tokens[-1]))

        # Convert range_tokens to a tensor if not already
        range_tokens_floats = [float(token) for token in self.range_tokens]
        range_tokens_tensor = torch.tensor(range_tokens_floats, device=floats.device, dtype=torch.float32)

        # Find the indices for each number using searchsorted
        positions = torch.searchsorted(range_tokens_tensor, floats_clipped, right=True) - 1
        positions = torch.clamp(positions, 0, len(self.range_tokens) - 2)  # Ensure indices are within the bounds

        # Lower and upper token values
        lower_tokens = range_tokens_tensor[positions]
        upper_tokens = range_tokens_tensor[positions + 1]

        # Calculate the interpolated indices
        lower_indices = self.float_token_id_start + positions

        # Compute the interpolated token indices
        fractional_part = (floats_clipped - lower_tokens) / (upper_tokens - lower_tokens)
        interpolated_indices = lower_indices + fractional_part

        return interpolated_indices

        
AutoConfig.register("videollama2_mistral", Videollama2MistralConfig)
AutoModelForCausalLM.register(Videollama2MistralConfig, Videollama2MistralForCausalLM)
