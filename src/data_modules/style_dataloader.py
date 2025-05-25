import json
import os
import random
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, ConcatDataset, Sampler
import lightning as L
import datasets
from datasets import load_dataset
import tqdm
from transformers import PreTrainedTokenizer

from src.data_modules.constants import ALL_PRETRAIN_DATA as ALL_DATA
from src.data_modules.style_dataset import PretrainStyleRepDataset, PretrainStyleRepCollator, num_proc
from src.data_modules.preprocess import preprocess_dataset


class InClusterDataSampler(Sampler):
    """
    A sampler for ConcatDataset that gurantees that each batch will comes from same dataset and same cluster.
    """ 
    def __init__(
            self,
            cluster_with_id: List[Dict[str, List[int]]],
            each_data_sizes: List[int],
            global_batch_size: int,
            shuffle: bool = True,
            num_replicas: int = 1,
            rank: int = 0,
            seed: int = 777,
            drop_last: bool = False,
            ):
        """
        :param each_data_sizes: list of sizes of each dataset
        :param global_batch_size: global batch size
        :param shuffle: whether to shuffle the indices
        :param num_replicas: number of replicas i.e. number of gpus
        :param rank: rank of the current gpu
        :param seed: seed for random number generator
        :param drop_last: whether to drop the last batch if it is incomplete
        """
        self.cluster_with_id = cluster_with_id
        self.each_data_sizes = each_data_sizes
        self.batch_size = global_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.indices = self.set_indices()
        self.num_samples = len(self.indices) // self.num_replicas

    def __iter__(self):
        # subsample
        indices = self.indices[self.rank:len(self.indices):self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    
    def set_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        rnd = random.Random(self.seed + self.epoch)
        assert len(self.cluster_with_id) == len(self.each_data_sizes), 'Number of datasets should be equal'
        indices = []
        for ds_with_cluster in self.cluster_with_id:
            _indices = []
            for _, in_cluster_ids in ds_with_cluster.items():
                in_cluster_ids = list(in_cluster_ids)
                if self.shuffle:
                    rnd.shuffle(in_cluster_ids)
                _indices.extend(in_cluster_ids)
            indices.append(_indices)

        # increase the indices by the offset
        assert len(indices) == len(self.each_data_sizes), 'Number of datasets should be equal'
        for i in range(len(self.each_data_sizes)):
            assert len(indices[i]) == self.each_data_sizes[i], 'Number of indices should be equal to the dataset size'
            indices[i] = [idx + sum(self.each_data_sizes[:i]) for idx in indices[i]]
        batched_indices = []
        for data_indices in indices:
            _batched_indices = list(torch.split(torch.tensor(data_indices), self.batch_size))
            batched_indices.append(_batched_indices)
        
        # Create separate batches from the remaining samples
        incomplete_indices = []
        for b in batched_indices:
            if len(b[-1]) < self.batch_size:
                incomplete_indices.append(b.pop())
        
        if self.drop_last is False and len(incomplete_indices) != 0:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=g).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(incomplete_indices, self.batch_size))
            if len(mixed_batches[-1]) < self.batch_size:
                mixed_batches.pop()
            batched_indices = sum(batched_indices, []) + mixed_batches
        else:
            batched_indices = sum(batched_indices, [])

        if self.shuffle:
            # Shuffle the batches 
            order = torch.randperm(len(batched_indices), generator=g).tolist()
        else:
            order = list(range(len(batched_indices)))
                         
        indices = []
        for batch_idx in order:
            indices.extend([int(i) for i in batched_indices[batch_idx]])
        return indices


class StyleDataModule(L.LightningDataModule):
    def __init__(
            self,
            seed: int=777,
            num_workers: int=1,
            preprocess_data_dir: str=None,
    ):
        super().__init__()
        self.seed = seed
        self.num_workers = num_workers
        self.datanames = ALL_DATA
        self.datanames.sort()
        self.preprocess_data_dir = preprocess_data_dir

    def connect(
            self, 
            model_type: str,
            tokenizer: PreTrainedTokenizer,
            world_size: int = 1,
            global_rank: int = 0,
            global_batch_size: int = 32,
            max_seq_length: int = 512,
            num_train_example: int = -1,
            num_positives: int = 1,
            num_hard_negatives: int = 256,
            ):
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.world_size = world_size
        self.global_rank = global_rank
        self.global_batch_size = global_batch_size
        self.max_seq_length = max_seq_length
        self.num_train_example = num_train_example
        self.num_positives = num_positives
        self.num_hard_negatives = num_hard_negatives

        self.batch_size = self.global_batch_size // self.world_size
        if self.global_batch_size % self.world_size != 0:
            print(f"Warning: global batch size {self.global_batch_size} is not divisible by world size {self.world_size}.")
            print(f"Setting batch size to {self.batch_size} and reducing global batch size to {self.batch_size * self.world_size}.")
            self.global_batch_size = self.batch_size * self.world_size
    
    def set_epoch(self, epoch: int) -> None:
        self.seed = self.seed + epoch
    
    def setup(self, model_checkpoint_dir, stage='') -> None:
        train_ds = []
        for dataname in self.datanames:
            if self.preprocess_data_dir is not None:
                dataname = os.path.join(self.preprocess_data_dir, dataname)
            dataset = PretrainStyleRepDataset(
                data_name_or_path=dataname,
                num_train_example=self.num_train_example,
                num_hard_negatives=self.num_hard_negatives,
                num_positives=self.num_positives,
                seed=self.seed,
            )
            if len(dataset) == 0:
                print(f'[Rank: {self.global_rank}]: No data loaded from {dataname}')
                continue
            train_ds.append(dataset)
            if self.global_rank == 0:
                print(f'Loaded {dataname} with {len(dataset)} samples')
        assert len(train_ds) > 0, 'No data loaded from the data files'
        self.train_ds = ConcatDataset(train_ds)

        # Load the author dict
        if os.path.exists(os.path.join(model_checkpoint_dir, 'author_dict.json')):
            with open(os.path.join(model_checkpoint_dir, 'author_dict.json'), 'r') as f:
                self.author_dict = json.load(f)
        else:
            raise ValueError('Please provide the author dict file to load the author dict')
    
    def train_dataloader(self) -> DataLoader:
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
        num_workers = min(self.num_workers, max_num_worker_suggest)
        collator = PretrainStyleRepCollator(
            tokenizer=self.tokenizer,
            author_id_dict=self.author_dict,
            max_seq_length=self.max_seq_length,
            model_type=self.model_type,
        )
        each_data_sizes = [len(dataset) for dataset in self.train_ds.datasets]
        cluster_infor = [dataset.cluster_info for dataset in self.train_ds.datasets]
        sampler = InClusterDataSampler(
                cluster_with_id=cluster_infor,
                each_data_sizes=each_data_sizes,
                global_batch_size=self.global_batch_size,
                shuffle=True,
                num_replicas=self.world_size,
                rank=self.global_rank,
                seed=self.seed,
                drop_last=False,
            )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collator,
        )
    
    def prepare_data(
            self, 
            use_dense_retrieval_hard_negatives: bool = True,
            style_encoder = None,
            content_encoder = None,
            num_clusters: int = None,
            threshold: float = None,
            do_filter: bool = False,
            ):
        """
        Download the dataset and do preprocessing like find the positive and negative samples and save them into disk
        Args:
            use_dense_retrieval_hard_negatives: whether to use dense retrieval hard negatives
            style_encoder: the style encoder model
            content_encoder: the content encoder model
            num_clusters: number of clusters to cluster the dataset
            threshold: threshold for filtering the documents
        """
        assert self.preprocess_data_dir is not None, 'Please provide the preprocess data directory to save the preprocessed data'
        for dataname in self.datanames:
            preprocess_dataset(
                dataname=dataname,
                use_dense_retrieval_hard_negatives=use_dense_retrieval_hard_negatives,
                style_encoder=style_encoder,
                content_encoder=content_encoder,
                num_clusters=num_clusters,
                threshold=threshold,
                do_filter=do_filter,
                preprocess_data_dir=self.preprocess_data_dir,
            )
        
    def get_the_author_dict(self, model_checkpoint_dir: str):
        """
        Get the author dict for all the datasets
        """
        all_author_ids = []
        for dataname in self.datanames:
            try:
                dataname = os.path.join(self.preprocess_data_dir, dataname)
                dataset = datasets.load_from_disk(dataname)
            except:
                dataset = datasets.load_dataset(dataname, split='train')
            dataset = dataset.map(
                lambda x: {'authorIDs': f"{dataname}-{x['authorIDs']}"}, 
                num_proc=num_proc,
                )
            author_ids = dataset['authorIDs']
            all_author_ids.extend(author_ids)
            dataset.cleanup_cache_files()
        all_author_ids = set(all_author_ids)
        author_dict = {idx: i for i, idx in enumerate(all_author_ids)}
        # Save the author dict to disk
        with open(os.path.join(model_checkpoint_dir, 'author_dict.json'), 'w') as f:
            json.dump(author_dict, f)
            
                    
if __name__ == '__main__':
    import yaml
    from dataclasses import asdict
    from transformers import AutoTokenizer, HfArgumentParser
    from src.args import (
        StyleEncoderDataArguments,
        EncoderModelArguments,
        EncoderTrainingArguments,
        )
    

    config_file = 'scripts/configs/style_encoder.yaml'
    hf_parser = HfArgumentParser((StyleEncoderDataArguments, EncoderModelArguments, EncoderTrainingArguments))
    data_args, model_args, training_args = hf_parser.parse_yaml_file(config_file)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    data_module = StyleDataModule(
            seed=training_args.seed,
            num_workers=training_args.num_workers,
            preprocess_data_dir=data_args.preprocess_data_dir,
        )
    data_module.connect(
            model_type=model_args.model_type,
            tokenizer=tokenizer,
            world_size=1,
            global_rank=0,
            global_batch_size=training_args.global_batch_size,
            max_seq_length=data_args.max_seq_length,
            num_train_example=data_args.number_of_training_samples,
            num_positives=data_args.num_positive,
            num_hard_negatives=data_args.num_hard_negative,
        )
    data_module.get_the_author_dict(model_checkpoint_dir=training_args.checkpoint_dir)
    data_module.setup(model_checkpoint_dir=training_args.checkpoint_dir)
    train_dataloader = data_module.train_dataloader()

    for batch in train_dataloader:
        breakpoint()
        
    
    
    
