from functools import partial
from typing import List, Optional, Tuple, Union
import einops
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RotaryEmbedding,
    Qwen3Model,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    QWEN3_START_DOCSTRING,
    QWEN3_INPUTS_DOCSTRING
)
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.utils import (
    logging,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

_CONFIG_FOR_DOC = "Qwen3Config"

logger = logging.get_logger(__name__)


class BidirectionalQwen3(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)

        # Check for _attn_implementation should be 'sdpa' or 'flash_attention_2'
        if config._attn_implementation not in ["sdpa", "flash_attention_2"]:
            config._attn_implementation = "flash_attention_2"
            logger.warning_once(
                f"Invalid attention implementation '{config._attn_implementation}'. Defaulting to 'sdpa'."
            )

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

         # Modified
        # Important: Set the causal flag to False for all attention layers (helpful for flash attention)
        for layer in self.layers:
            assert hasattr(layer, "self_attn"), "Layer must have self_attn module"
            assert hasattr(layer.self_attn, "is_causal"), "self_attn module must have is_causal attribute"
            layer.self_attn.is_causal = False

        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_attention_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_attention_mask(
            self, 
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool = False,
            ) -> torch.Tensor:
        
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    logger.warning_once(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        
        if output_attentions:
            if self.config._attn_implementation != "eager":
                logger.warning_once(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`.'
                    " Setting `output_attentions=False`."
                )
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        
        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        sequence_length: int = input_tensor.shape[1]
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length: int = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        attend_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            ) # Full of min_dtype
        bidirectional_attend = torch.full(
                (sequence_length, target_length), fill_value=True, dtype=dtype, device=device
        )
        if self.config.sliding_window is not None:
            # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
            # the check is needed to verify is current checkpoint was trained with sliding window or not
            if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                sliding_attend = torch.abs(torch.arange(target_length, device=device) - cache_position.reshape(-1, 1)) <= self.config.sliding_window // 2 # [sequence_length, target_length]
                bidirectional_attend.logical_and_(sliding_attend) # i.e., True for tokens within sliding window, False for tokens outside
        bidirectional_attend.logical_not_() # i.e., False for tokens attending, True for tokens not attending
        attend_mask *= bidirectional_attend # i.e., 0 for tokens attending, min_dtype for tokens not attending
        attend_mask = attend_mask[None, None, :, :].expand(input_tensor.shape[0], 1, sequence_length, target_length)
        if attention_mask is not None:
            attend_mask = attend_mask.clone() 
            if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = attend_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                attend_mask.device
            )
            # Find the remaining not masked positions (i.e., 0) and set them to min_dtype
            padding_mask = padding_mask == 0
            attend_mask[:, :, :, :mask_length] = attend_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return attend_mask

