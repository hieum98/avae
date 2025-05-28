import os
from typing import List, Optional, Tuple, Union
from einops import reduce, repeat
import numpy as np
import torch
import torch.nn as nn
import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer, 
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from huggingface_hub import PyTorchModelHubMixin

from src.model.qwen2_bidirectional import BidirectionalQwen2
from src.model.qwen3_bidirectional import BidirectionalQwen3
from src.model.vae import VAE, VQVAE
from src.model.utils import find_all_linear_names
from src.data_modules.templates import apply_template_for_rep_learning, tokenize_example


class Encoder(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            model_name_or_path: str,
            embedding_dim: int = 1536,
            use_lora: bool = False,
            trainable: bool = False,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            target_modules: Union[str, List[str]] = "all",
            adapter_name: str = None,
            attn_implementation: str = None,
            pooling_method: str = 'mean',
            dropout_prob: float = 0.1,
            model_type: str = 'qwen2',
            use_bidirectional: bool = True,
            use_vae: bool = True,
    ):
        super(Encoder, self).__init__()
        if use_bidirectional:
            assert model_type in ['qwen2', 'qwen3'], "Only Qwen supports bidirectional model at the moment."
        self.hprams = {
            'model_name_or_path': model_name_or_path,
            'trainable': trainable,
            'use_lora': use_lora,
            'dropout_prob': dropout_prob,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules,
            'adapter_name': adapter_name,
            'attn_implementation': attn_implementation,
            'pooling_method': pooling_method,
            'model_type': model_type,
            'use_bidirectional': use_bidirectional,
            'use_vae': use_vae,
            'embedding_dim': embedding_dim,
        }   
        self.model_type = model_type
        self.pooling_method = pooling_method
        self.use_vae = use_vae

        self.transformer: PreTrainedModel = self.load_model(
            model_name_or_path,
            use_bidirectional=use_bidirectional,
            model_type=model_type,
            trainable=trainable,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            attn_implementation=attn_implementation,
            adapter_name=adapter_name
        )
        self.tokenizer: PreTrainedTokenizer = self.create_tokenizer(model_name_or_path)

        self.dropout = nn.Dropout(dropout_prob)
        self.projection = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, self.transformer.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.transformer.config.hidden_size, embedding_dim),
        )
        self.emb_dim = embedding_dim

        if self.use_vae:
            self.vae = VAE(
                input_dim=self.emb_dim,
                latent_dim=self.emb_dim,
            )

    @staticmethod
    def load_model(
            model_name_or_path: str, 
            use_bidirectional: bool = True,
            model_type: str = 'qwen2',
            trainable: bool = False,
            use_lora: bool = False, 
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
        if use_bidirectional:
            if model_type == 'qwen2':
                # Load the bi-directional Qwen2 model
                print("Loading bi-directional Qwen2 model...")
                transformer: PreTrainedModel = BidirectionalQwen2.from_pretrained(
                    model_name_or_path,
                    config=config,
                    attn_implementation=attn_implementation,
                    torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
                )
            elif model_type == 'qwen3':
                # Load the bi-directional Qwen3 model
                print("Loading bi-directional Qwen3 model...")
                transformer: PreTrainedModel = BidirectionalQwen3.from_pretrained(
                    model_name_or_path,
                    config=config,
                    attn_implementation=attn_implementation,
                    torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
                )
        else:
            print("Loading AutoModel...")
            transformer: PreTrainedModel = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
            )

        if use_lora:
            if target_modules == "all":
                target_modules = find_all_linear_names(transformer)
            assert isinstance(target_modules, list) or target_modules == 'all-linear', "target_modules must be a list or 'all-linear'"
            task_type = TaskType.FEATURE_EXTRACTION
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
        elif trainable is False:
            # Freeze the model
            for param in transformer.parameters():
                param.requires_grad = False
        return transformer
    
    @staticmethod
    def create_tokenizer(model_name_or_path: str):
        # Load tokenizer
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        tokenizer.padding_side = 'right' # Set padding side to right in all cases
        if tokenizer.pad_token_id is None:
            print("Tokenizer does not have a pad token. We will use the eos token as pad token.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def pooling(
            self,
            hidden_state: torch.Tensor, # [batch_size, seq_len, hidden_size]
            attention_mask: torch.Tensor = None, # [batch_size, seq_len]
            prompt_length: Optional[torch.Tensor] = None, # [batch_size]
        ):  
        if attention_mask is None:
            attention_mask = torch.ones(hidden_state.size(0), hidden_state.size(1), device=hidden_state.device)
        # Pool the hidden states
        # Mask the prompt tokens
        if prompt_length is not None:
            attention_mask = attention_mask.clone()
            for i, l in enumerate(prompt_length):
                if self.tokenizer.padding_side == 'right':
                    attention_mask[i, :l] = 0
                elif self.tokenizer.padding_side == 'left':
                    # get the first index of the unmasked tokens
                    start_index = attention_mask[i].nonzero().min()
                    attention_mask[i, start_index:start_index + l] = 0
                # Make sure not all zeros - If this happens it is a bug
                assert attention_mask[i].sum() > 0, "You have all zeros in the attention mask!"

        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)
        if self.pooling_method == 'cls':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
                # Get the first index of the unmasked tokens
                embedding = hidden_state[torch.arange(batch_size, device=hidden_state.device), -1*sequence_lengths]
            else:
                embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                embedding = hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_state.shape[0]
                embedding = hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]
        elif self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        else: raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")
        
        return embedding.contiguous().to(hidden_state.dtype)

    def forward(
            self,
            input_ids: torch.Tensor, # [batch_size, seq_len]
            attention_mask: torch.Tensor, # [batch_size, seq_len]
            prompt_length: Optional[torch.Tensor] = None, # [batch_size]
    ):
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        # TODO: Use multiple layers' hidden states
        hidden_states = output['hidden_states'][-1]  # [batch_size, seq_len, hidden_size]

        pooled_output = self.pooling(hidden_states, attention_mask, prompt_length)
        projection_output = self.projection(self.dropout(pooled_output)) # (batch_size, hidden_size)
        if self.use_vae:
            vae_outputs = self.vae(input=projection_output)
            if self.training:
                reps = vae_outputs['z'] # [batch_size, embedding_dim]
                mu = vae_outputs['mu'] # [batch_size, embedding_dim]
                # logvar = vae_outputs['logvar'] # [batch_size, embedding_dim]
            else:
                reps = vae_outputs['mu']
                mu = vae_outputs['mu']
                # logvar = vae_outputs['logvar']
            # TODO: try to use concatenation of mu and var
            # reps = torch.cat([vae_outputs['mu'], vae_outputs['logvar']], dim=-1)
            # TODO: try to use the reparameterization trick
            # reps = vae_outputs['z']
            vae_loss = vae_outputs['kld_loss']
        else:
            reps = projection_output
            mu = projection_output
            vae_loss = None
            # logvar = None
        return {
            'reps': reps, # [batch_size, embedding_dim]
            'mu': mu, # [batch_size, embedding_dim]
            # 'logvar': logvar, # [batch_size, embedding_dim]
            'vae_loss': vae_loss,
            'hidden_states': hidden_states, # [batch_size, seq_len, hidden_size]
        }
    
    def to(self, device: Union[str, torch.device]):
        super(Encoder, self).to(device)
        if device == 'cuda' or device == torch.device('cuda'):
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    @torch.no_grad()
    def encode(
            self,
            texts: List[str],
            max_length: int = 512,
            batch_size: int = 32,
            return_hidden_states: bool = False,
            **kwargs
            ):
        self.eval()
        all_reps = []
        all_hidden_states = []
        if hasattr(self, 'device'):
            device = self.device
        else:
            device = 'cpu'
        for i in tqdm.tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_texts = apply_template_for_rep_learning(batch_texts, self.model_type)
            inputs = tokenize_example(
                batch_texts,
                self.tokenizer,
                max_seq_length=max_length,
                truncation=True,
                padding='longest',
                return_tensors='pt',
            )
            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    output = self(input_ids, attention_mask)
                    reps = output['reps']
                    # Mask out the padding tokens
                    hidden_states = output['hidden_states'] * output['attention_mask'].unsqueeze(-1)  # [batch_size, seq_len, hidden_size]
            else:
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                output = self(input_ids, attention_mask)
                reps = output['reps']
                hidden_states = output['hidden_states'] * output['attention_mask'].unsqueeze(-1)  # Mask out the padding tokens
            all_reps.append(reps.cpu())
            all_hidden_states.append(hidden_states.cpu())
        all_reps = torch.cat(all_reps, dim=0)
        max_seq_len = max([x.shape[1] for x in all_hidden_states])
        padded_all_hidden_states = torch.zeros((all_reps.size(0), max_seq_len, all_hidden_states[0].size(-1)))
        i = 0
        for hidden_states in all_hidden_states:
            bs, seq_len, hidden_size = hidden_states.size()
            padded_all_hidden_states[i:i + bs, :seq_len, :] = hidden_states
            i += bs
        return all_reps if not return_hidden_states else (all_reps, padded_all_hidden_states)
    

class WrappedEncoder(nn.Module):
    def __init__(
            self, 
            encoder: nn.Module,
            tokenizer: PreTrainedTokenizer = None,
            model_checkpoint: Optional[str] = None,
            num_gpus: int = 1,
            gpu_id: Optional[int] = None,
            ):
        super(WrappedEncoder, self).__init__()
        self.encoder = encoder
        if isinstance(encoder, Encoder):
            self.tokenizer = encoder.tokenizer
            self.model_type = encoder.model_type
        else:
            self.tokenizer = tokenizer
            self.model_type = ''
        if model_checkpoint is not None and os.path.exists(model_checkpoint):
            print(f"Loading model from checkpoint: {model_checkpoint}")
            state_dict = torch.load(model_checkpoint, map_location='cpu')
            imcomplete_keys = self.encoder.load_state_dict(state_dict['model'], strict=False)
            print(f"Loaded model with missing keys: {imcomplete_keys}")
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None else 'cuda')
            self.is_cuda = True
        else:
            self.device = torch.device('cpu')
            self.is_cuda = False
        self.num_gpus = min(torch.cuda.device_count(), num_gpus)
        print(f"Using {self.num_gpus} GPUs")
        self.encoder.to(self.device)
        if self.num_gpus > 1:
            self.encoder = nn.DataParallel(self.encoder)
        self.encoder.eval()

    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(
            self,
            input_ids: torch.Tensor,  # [batch_size, seq_len]
            attention_mask: torch.Tensor,  # [batch_size, seq_len]
            prompt_length: Optional[torch.Tensor] = None,  # [batch_size]
        ):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if self.is_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                output = self.encoder(input_ids, attention_mask,)
        else:
            with torch.autocast(device_type='cpu', dtype=torch.float32):
                output = self.encoder(input_ids, attention_mask,)
        if isinstance(self.encoder, Encoder):
            reps = output['reps']
            hidden_states = output.get('hidden_states', None)
        else:
            hidden_states = output['last_hidden_state'] * attention_mask.unsqueeze(-1)
            reps = self.mean_pooling(hidden_states, attention_mask)
        return {
            'reps': reps,  # [batch_size, embedding_dim]
            'hidden_states': hidden_states,  # [batch_size, seq_len, hidden_size]
        }
    
    @torch.no_grad()
    def encode(
            self,
            texts: List[str],
            max_length: int = 512,
            batch_size: int = 32,
            return_hidden_states: bool = False,
            **kwargs
            ):
        all_reps = []
        all_hidden_states = []
        device = self.device
        batch_size = batch_size * self.num_gpus
        for i in tqdm.tqdm(range(0, len(texts), batch_size), disable=len(texts) < 10000):
            batch_texts = texts[i:i + batch_size]
            batch_texts = apply_template_for_rep_learning(batch_texts, self.model_type)
            inputs = tokenize_example(
                batch_texts,
                self.tokenizer,
                max_seq_length=max_length,
                truncation=True,
                padding='longest',
                return_tensors='pt',
            )
            if self.is_cuda:
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    output = self.encoder(input_ids, attention_mask)
                    if 'reps' in output:
                        reps = output['reps']
                        hidden_states = output.get('hidden_states', None)
                        if hidden_states is not None:
                            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                    else:
                        hidden_states = output['last_hidden_state'] * attention_mask.unsqueeze(-1)
                        reps = self.mean_pooling(hidden_states, attention_mask)
            else:
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                output = self.encoder(input_ids, attention_mask)
                if 'reps' in output:
                    reps = output['reps']
                    hidden_states = output.get('hidden_states', None)
                    if hidden_states is not None:
                        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                else:
                    hidden_states = output['last_hidden_state'] * attention_mask.unsqueeze(-1)
                    reps = self.mean_pooling(hidden_states, attention_mask)
            all_reps.append(reps.cpu())
            all_hidden_states.append(hidden_states.cpu() if hidden_states is not None else None)
        all_reps = torch.cat(all_reps, dim=0)
        if any(x is None for x in all_hidden_states):
            all_hidden_states = None
        else:
            max_seq_len = max([x.shape[1] for x in all_hidden_states])
            padded_all_hidden_states = torch.zeros((all_reps.size(0), max_seq_len, all_hidden_states[0].size(-1)), device=device)
            i = 0
            for hidden_states in all_hidden_states:
                bs, seq_len, hidden_size = hidden_states.size()
                padded_all_hidden_states[i:i + bs, :seq_len, :] = hidden_states
                i += bs
            all_hidden_states = padded_all_hidden_states

        return all_reps if not return_hidden_states else (all_reps, padded_all_hidden_states)


if __name__ == '__main__':
    model_name_or_path = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
    embedding_dim = 1536
    use_lora = False
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    target_modules = "all"
    adapter_name = None
    attn_implementation = 'flash_attention_2'
    pooling_method = 'lasttoken'
    dropout_prob = 0.1
    model_type = 'qwen2'
    use_bidirectional = False

    encoder = Encoder(
        model_name_or_path=model_name_or_path,
        embedding_dim=embedding_dim,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        adapter_name=adapter_name,
        attn_implementation=attn_implementation,
        pooling_method=pooling_method,
        dropout_prob=dropout_prob,
        model_type=model_type,
        use_bidirectional=use_bidirectional,
    )
    encoder.to('cuda')
    txt = ["Hello, world!", "This is a great day to code."]
    emb = encoder.encode(txt)
    breakpoint()
