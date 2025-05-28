import json
import os
import random
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
import lightning as L
import datasets
from transformers import PreTrainedTokenizer

from src.data_modules.disentanglement_dataset import DisentanglementDataset, DisentanglementDatasetCollator, num_proc
from src.data_modules.constants import ALL_DISENT_DATA as ALL_DATA


class DisentanglementDataModule(L.LightningDataModule):
    def __init__(
        self,
        seed: int=777,
        num_workers: int=1,
        ):
        super().__init__()
        self.seed = seed
        self.num_workers = num_workers
        self.datanames = ALL_DATA
        self.datanames.sort()

    def connect(
        self,
        style_encoder_tokenizer: PreTrainedTokenizer,
        content_encoder_tokenizer: PreTrainedTokenizer,
        generator_tokenizer: PreTrainedTokenizer, 
        max_length: int,
        global_batch_size: int = 32,
        world_size: int = 1,
        global_rank: int = 0,
        num_train_example: int = 100000,
        placeholder_token: str = " <|placeholder|> ",
        prompt_loss = False,
        ):
        self.style_encoder_tokenizer = style_encoder_tokenizer
        self.content_encoder_tokenizer = content_encoder_tokenizer
        self.generator_tokenizer = generator_tokenizer
        self.max_length = max_length
        self.global_batch_size = global_batch_size
        self.world_size = world_size
        self.global_rank = global_rank
        self.num_train_example = num_train_example
        self.prompt_loss = prompt_loss
        
        self.placeholder_token = placeholder_token
        # If placeholder is not in the tokenizer, add it
        if self.placeholder_token not in self.generator_tokenizer.get_vocab():
            self.generator_tokenizer.add_tokens([self.placeholder_token])
            # Get the token id of the placeholder
            self.placeholder_token_id = self.generator_tokenizer.convert_tokens_to_ids(self.placeholder_token)

        self.batch_size = self.global_batch_size // self.world_size
        if self.global_batch_size % self.world_size != 0:
            print(f"Warning: global batch size {self.global_batch_size} is not divisible by world size {self.world_size}.")
            print(f"Setting batch size to {self.batch_size} and reducing global batch size to {self.batch_size * self.world_size}.")
            self.global_batch_size = self.batch_size * self.world_size
    
    def set_epoch(self, epoch: int) -> None:
        self.seed = self.seed + epoch

    def setup(self, **kwargs) -> None:
        train_datasets = []
        for name in self.datanames:
            dataset = DisentanglementDataset(
                data_name_or_path=name,
                tokenizer=self.generator_tokenizer,
                num_train_example=self.num_train_example,
                placeholder=self.placeholder_token,
                seed=self.seed,
            )
            style_encoder = kwargs.get('style_encoder', None)
            content_encoder = kwargs.get('content_encoder', None)
            cache_name = kwargs.get('cache_name', 'precomputed_hidden')
            if style_encoder is not None and content_encoder is not None:
                dataset.precompute_ref_representation(style_encoder=style_encoder, content_encoder=content_encoder, cache_name=cache_name)
            if self.global_rank == 0:
                print(f"Loaded dataset {name} with {len(dataset)} examples.")
            if len(dataset) > 0:
                train_datasets.append(dataset)
        if len(train_datasets) > 0:
            self.train_dataset = ConcatDataset(train_datasets)
        else:
            raise ValueError("No datasets found. Please check the dataset paths and names.")

    def train_dataloader(self) -> DataLoader:
        num_workers = min(self.num_workers, num_proc)
        collator = DisentanglementDatasetCollator(
            style_encoder_tokenizer=self.style_encoder_tokenizer,
            content_encoder_tokenizer=self.content_encoder_tokenizer,
            generator_tokenizer=self.generator_tokenizer,
            max_length=self.max_length,
            placeholder_token_id=self.placeholder_token_id,
            placeholder=self.placeholder_token,
            prompt_loss=self.prompt_loss,
        )
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True,
            seed=self.seed,
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collator,
            num_workers=num_workers,
            sampler=sampler,
        )
        return dataloader
    

if __name__=='__main__':
    # Example usage
    from transformers import AutoTokenizer, AutoModel
    import hashlib
    from src.model.encoder import WrappedEncoder

    encoder_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    generator_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    style_encoder_name = "Hieuman/qwen2-1.5b-hard-author-reps"
    content_encoder_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    style_encoder = AutoModel.from_pretrained(style_encoder_name)
    style_tokenizer = AutoTokenizer.from_pretrained(style_encoder_name)
    content_encoder = AutoModel.from_pretrained(content_encoder_name)
    content_tokenizer = AutoTokenizer.from_pretrained(content_encoder_name)
    style_encoder = WrappedEncoder(style_encoder, style_tokenizer, num_gpus=8)
    content_encoder = WrappedEncoder(content_encoder, content_tokenizer, num_gpus=8)
    cache_name = f"style:{style_encoder_name}_content:{content_encoder_name}"
    # hash the cache name to get a unique identifier
    cache_name = hashlib.md5(cache_name.encode()).hexdigest()
    print(f"Using cache name: {cache_name}")

    dataloader = DisentanglementDataModule(
        seed=777,
        num_workers=0,
    )
    dataloader.connect(
        style_encoder_tokenizer=encoder_tokenizer,
        content_encoder_tokenizer=encoder_tokenizer,
        generator_tokenizer=generator_tokenizer,
        max_length=2048,
        global_batch_size=8,
        world_size=1,
        global_rank=0,
        num_train_example=100000000,  # Use a large number to load all examples
        placeholder_token=" <|placeholder|> ",
        prompt_loss=False,
    )
    dataloader.setup(style_encoder=style_encoder, content_encoder=content_encoder, cache_name=cache_name)
    train_dataloader = dataloader.train_dataloader()
    for batch in train_dataloader:
        breakpoint()


