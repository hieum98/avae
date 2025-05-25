import os
from typing import List, Optional, Tuple, Union
from einops import reduce, repeat
import numpy as np
import torch
import torch.nn as nn
import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationMixin,
)
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from huggingface_hub import PyTorchModelHubMixin

from src.model.utils import find_all_linear_names


class Generator(nn.Module, GenerationMixin, PyTorchModelHubMixin):
    def __init__(
        self,
        model_name_or_path: str,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Union[str, List[str]] = "all",
        adapter_name: str = None,
        attn_implementation: str = None,
        input_embedding_dim: int = None,
    ):
        super().__init__()
        self.hprams = {
            "model_name_or_path": model_name_or_path,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "adapter_name": adapter_name,
            "attn_implementation": attn_implementation,
        }

        self.tranformer: PreTrainedModel = self.load_model(
            model_name_or_path,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            attn_implementation=attn_implementation,
            adapter_name=adapter_name
        )
        self.tokenizer: PreTrainedTokenizer = self.load_tokenizer(
            model_name_or_path,
            trust_remote_code=True,   
        )
        if input_embedding_dim is not None:
            self.connector: nn.Sequential = nn.Sequential(
                nn.Linear(input_embedding_dim, self.tranformer.config.hidden_size),
                nn.SELU(),
                nn.Linear(self.tranformer.config.hidden_size, self.tranformer.config.hidden_size),
            )
    
    @staticmethod
    def load_model(
            model_name_or_path: str, 
            use_lora: bool = False, 
            trainable: bool = False,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            target_modules: Union[str, List[str]] = "all",
            attn_implementation: str = None,
            adapter_name: Optional[str] = None
        ) -> PreTrainedModel:
        if use_lora:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                pretraining_tp=1,  # Fix mat1 and mat2 shapes cannot be multiplied  error with LLaMA-2
                # See https://github.com/huggingface/transformers/pull/24906
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False
            )
        transformer: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
        )

        if use_lora:
            if target_modules == "all":
                target_modules = find_all_linear_names(transformer)
            assert isinstance(target_modules, list) or target_modules == 'all-linear', "target_modules must be a list or 'all-linear'"
            task_type = TaskType.CAUSAL_LM
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=task_type,
                target_modules=target_modules,
            )
            if adapter_name is None:
                adapter_name = 'default'
            transformer: PeftModel = get_peft_model(transformer, lora_config, adapter_name=adapter_name)
        elif trainable==False:
            # Freeze the model
            for param in transformer.parameters():
                param.requires_grad = False
        return transformer

    @staticmethod
    def load_tokenizer(
        model_name_or_path: str,  
        **kwargs
    ) -> PreTrainedTokenizer:
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **kwargs
        )
        tokenizer.padding_side = "left" # Always pad on the left side 
        if tokenizer.pad_token_id is None:
            print("Tokenizer does not have a pad token. We will use the eos token as pad token.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    
    def forward(
        self, 
        input_ids: torch.LongTensor, # [batch_size, seq_len]
        attention_mask: Optional[torch.LongTensor], # [batch_size, seq_len]
        labels: Optional[torch.LongTensor] = None, # [batch_size, seq_len]
        first_sentence_embedding: Optional[torch.Tensor] = None, # [batch_size, hidden_size]
        second_sentence_embedding: Optional[torch.Tensor] = None, # [batch_size, hidden_size]
        # Tuple of (tensor, tensor) with the positions of the placeholder tokens, with the first tensor being the batch index and the second tensor being the token index
        placeholder_token_pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if placeholder_token_pos is not None and first_sentence_embedding is not None and second_sentence_embedding is not None:
            assert placeholder_token_pos[0].size(0) == input_ids.size(0) * 2, "The number of placeholder tokens must be equal to the batch size times 2"
            assert first_sentence_embedding.size(0) == input_ids.size(0), "First sentence embedding must be equal to the batch size"
            assert second_sentence_embedding.size(0) == input_ids.size(0), "The second sentence embedding must be equal to the batch size"
            inputs_embeds = self.tranformer.get_input_embeddings()(input_ids) # [batch_size, seq_len, hidden_size]
            if hasattr(self, "connector"):
                first_sentence_embedding = self.connector(first_sentence_embedding)
                second_sentence_embedding = self.connector(second_sentence_embedding)
            # Replace the placeholder tokens with the sentence embeddings
            # Do the mask operation to make sure gradient can flow through the sentence embeddings
            original_mask = torch.ones_like(inputs_embeds)
            original_mask[placeholder_token_pos[0], placeholder_token_pos[1], :] = 0
            # the placeholder tokens position present in position like [first_sentence, second_sentence, first_sentence, second_sentence, ...]
            first_placeholder_batch_index = placeholder_token_pos[0][::2]
            first_placeholder_token_index = placeholder_token_pos[1][::2] 
            first_placeholder_mask = torch.zeros_like(inputs_embeds)
            first_placeholder_mask[first_placeholder_batch_index, first_placeholder_token_index, :] = 1
            second_placeholder_batch_index = placeholder_token_pos[0][1::2]
            second_placeholder_token_index = placeholder_token_pos[1][1::2]
            second_placeholder_mask = torch.zeros_like(inputs_embeds)
            second_placeholder_mask[second_placeholder_batch_index, second_placeholder_token_index, :] = 1
            inputs_embeds = inputs_embeds * original_mask + first_placeholder_mask * first_sentence_embedding.unsqueeze(1) + second_placeholder_mask * second_sentence_embedding.unsqueeze(1)
            if labels is not None:
                # Replace the placeholder tokens with -100 in the labels to ignore them in the loss
                labels[placeholder_token_pos[0], placeholder_token_pos[1]] = -100
        else:
            inputs_embeds = self.tranformer.get_input_embeddings()(input_ids)

        outputs = self.tranformer(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
        )
        return_dict = {
            "loss": outputs.loss, 
            "logits": outputs.logits, # [batch_size, seq_len, vocab_size]
            "hidden_states": outputs.hidden_states, # List of [batch_size, seq_len, hidden_size]
        }
        if placeholder_token_pos is not None:
            first_placeholder_batch_index = placeholder_token_pos[0][::2]
            first_placeholder_token_index = placeholder_token_pos[1][::2]
            first_placeholder_token_logits = outputs.logits[first_placeholder_batch_index, first_placeholder_token_index, :]
            first_placeholder_token_reps = outputs.hidden_states[-1][first_placeholder_batch_index, first_placeholder_token_index, :]
            second_placeholder_batch_index = placeholder_token_pos[0][1::2]
            second_placeholder_token_index = placeholder_token_pos[1][1::2]
            second_placeholder_token_logits = outputs.logits[second_placeholder_batch_index, second_placeholder_token_index, :]
            second_placeholder_token_reps = outputs.hidden_states[-1][second_placeholder_batch_index, second_placeholder_token_index, :]
            return_dict.update({
                "first_placeholder_token_logits": first_placeholder_token_logits, # [batch_size, vocab_size]
                "first_placeholder_token_reps": first_placeholder_token_reps, # [batch_size, hidden_size]
                "second_placeholder_token_logits": second_placeholder_token_logits,
                "second_placeholder_token_reps": second_placeholder_token_reps,
            })
        return return_dict
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor, # [batch_size, seq_len]
        attention_mask: Optional[torch.LongTensor], # [batch_size, seq_len]
        first_sentence_embedding: Optional[torch.Tensor] = None, # [batch_size, hidden_size]
        second_sentence_embedding: Optional[torch.Tensor] = None, # [batch_size, hidden_size]
        placeholder_token_pos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **generation_kwargs,
    ) -> torch.LongTensor:
        """
        Override the generate method of the GenerationMixin class to be able to use the model as conditional generator based on the sentence embeddings.
        Args:
            input_ids (torch.LongTensor): The input ids of the text to be generated.
            attention_mask (torch.LongTensor): The attention mask of the input ids.
            first_sentence_embedding (torch.Tensor): The sentence embedding of the first sentence.
            second_sentence_embedding (torch.Tensor): The sentence embedding of the second sentence.
            placeholder_token_pos (Tuple[torch.Tensor, torch.Tensor]): The positions of the placeholder tokens in the input ids.
            generation_kwargs (dict): The generation arguments to be passed to the generate method.
        Returns:
            HuggingFace generate output
        """
        if placeholder_token_pos is not None and first_sentence_embedding is not None and second_sentence_embedding is not None:
            assert placeholder_token_pos[0].size(0) == input_ids.size(0) * 2, "The number of placeholder tokens must be equal to the batch size times 2"
            assert first_sentence_embedding.size(0) == input_ids.size(0), "First sentence embedding must be equal to the batch size"
            assert second_sentence_embedding.size(0) == input_ids.size(0), "The second sentence embedding must be equal to the batch size"
            inputs_embeds = self.tranformer.get_input_embeddings()(input_ids)
            if hasattr(self, "connector"):
                first_sentence_embedding = self.connector(first_sentence_embedding)
                second_sentence_embedding = self.connector(second_sentence_embedding)
            # Replace the placeholder tokens with the sentence embeddings
            # Do the mask operation to make sure gradient can flow through the sentence embeddings
            original_mask = torch.ones_like(inputs_embeds)
            original_mask[placeholder_token_pos[0], placeholder_token_pos[1], :] = 0
            # the placeholder tokens position present in position like [first_sentence, second_sentence, first_sentence, second_sentence, ...]
            first_placeholder_batch_index = placeholder_token_pos[0][::2]
            first_placeholder_token_index = placeholder_token_pos[1][::2]
            first_placeholder_mask = torch.zeros_like(inputs_embeds)
            first_placeholder_mask[first_placeholder_batch_index, first_placeholder_token_index, :] = 1
            second_placeholder_batch_index = placeholder_token_pos[0][1::2]
            second_placeholder_token_index = placeholder_token_pos[1][1::2]
            second_placeholder_mask = torch.zeros_like(inputs_embeds)
            second_placeholder_mask[second_placeholder_batch_index, second_placeholder_token_index, :] = 1
            inputs_embeds = inputs_embeds * original_mask + first_placeholder_mask * first_sentence_embedding.unsqueeze(1) + second_placeholder_mask * second_sentence_embedding.unsqueeze(1)
        else:
            inputs_embeds = self.tranformer.get_input_embeddings()(input_ids)
        
        inputs = {
            "input_ids": None,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        outputs = self.tranformer.generate(**inputs, **generation_kwargs)
        return outputs
        
        
if __name__=='__main__':

    from src.data_modules.disentanglememt_dataloader import DisentanglementDataModule

    model_name_or_path: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Union[str, List[str]] = "all"
    adapter_name: str = None
    attn_implementation: str = 'flash_attention_2'
    input_embedding_dim: int = 1536

    generator = Generator(
        model_name_or_path=model_name_or_path,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        adapter_name=adapter_name,
        attn_implementation=attn_implementation,
        input_embedding_dim=input_embedding_dim
    )
    generator.to("cuda", dtype=torch.bfloat16)
    tokenizer = generator.tokenizer
    dataloader = DisentanglementDataModule(
        seed=777,
        num_workers=0,
    )
    dataloader.connect(
        encoder_tokenizer=tokenizer,
        generator_tokenizer=tokenizer,
        max_length=2048,
        global_batch_size=8,
        world_size=1,
        global_rank=0,
        num_train_example=1000,
        placeholder_token=" <|placeholder|> ",
        prompt_loss=False,
    )
    dataloader.setup()
    train_dataloader = dataloader.train_dataloader()
    for batch in train_dataloader:
        input_ids = batch['txt_reconstruct_tokenized']['input_ids'].to("cuda")
        attention_mask = batch['txt_reconstruct_tokenized']['attention_mask'].to("cuda")
        labels = batch['txt_reconstruct_tokenized']['labels'].to("cuda")
        txt_placeholder_token_pos = batch['txt_placeholder_token_pos']
        bs = input_ids.size(0)
        first_sentence_embedding = torch.randn(bs, input_embedding_dim).to("cuda", dtype=torch.bfloat16)
        second_sentence_embedding = torch.randn(bs, input_embedding_dim).to("cuda", dtype=torch.bfloat16)
        model_outputs = generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            first_sentence_embedding=first_sentence_embedding,
            second_sentence_embedding=second_sentence_embedding,
            placeholder_token_pos=txt_placeholder_token_pos
        )

        # Generate
        generation_kwargs = {
            "max_length": 2096,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.8,
            "num_return_sequences": 1,
            "eos_token_id": tokenizer.eos_token_id,
        }
        generated_ids = generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            first_sentence_embedding=first_sentence_embedding,
            second_sentence_embedding=second_sentence_embedding,
            placeholder_token_pos=txt_placeholder_token_pos,
            **generation_kwargs
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        breakpoint()



