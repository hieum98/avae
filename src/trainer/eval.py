from glob import glob
import json
import os
import pathlib
import datasets
import faiss
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict, concatenate_datasets

from src.data_modules.templates import tokenize_example
from src.data_modules.utils import flatten, get_embedding, group_by_column, simple_preprocess
from src.model.encoder import Encoder, WrappedEncoder
from src.model.generator import Generator
from src.data_modules.constants import HRS_PATHS, AMAZON_REVIEWS_PATHS, MUD_PATHS, PAN20_PATHS, PAN21_PATHS


class WrappedLUAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.encoder.to(self.device)
        self.encoder.eval()
    
    @torch.no_grad()
    def encode(
            self,
            texts: List[str],
            max_length: int = 512,
            **kwargs
            ):
        device = self.device
        inputs = tokenize_example(
            texts,
            self.tokenizer,
            max_seq_length=max_length,
            truncation=True,
            padding='longest',
            return_tensors='pt',
        ) # [n_texts, max_length]
        if device == 'cuda' or device == torch.device('cuda'):
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                inputs['input_ids'] = inputs['input_ids'].unsqueeze(0).to(device) # [1, n_texts, max_length]
                inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0).to(device)
                rep = self.encoder(**inputs)
        else:
            rep = self.encoder(**inputs)
        return rep



def extract_author_embeddings(
        model,
        dataset: Dataset,
        ):
    """
    Extract author embeddings from the model for the given dataset.
    Args:
        model (Encoder): The encoder model that have the `encode` method that accepts a list of strings and returns a list of embeddings.
        dataset (Dataset): The dataset to extract embeddings from with the format {'authorIDs': str, 'fullText': List[str]}
    Returns:
        Dataset: The dataset with the embeddings added as a new column 'embeddings'.
    """
    assert hasattr(model, 'encode'), 'model should have the method encode to get embeddings'
    def get_embedding(model, texts):
        return model.encode(texts).squeeze(0).cpu()
    
    dataset = dataset.map(
        lambda x: {'embeddings': get_embedding(model, x['fullText'])},
        remove_columns=['fullText'],
        num_proc=None,
        desc="Extracting author embeddings"
    )
    return dataset


def extract_embeddings(        
        model: Encoder,
        dataset: Dataset,
        batch_size: int = 32
) -> Dataset:
    """
    Extract embeddings from the model for the given dataset.
    Args:
        model (Encoder): The encoder model that have the `encode` method that accepts a list of strings and returns a list of embeddings.
        dataset (Dataset): The dataset to extract embeddings from with the format {'authorIDs': str, 'documentID': str, 'fullText': str}
        batch_size (int): The batch size to use for extracting embeddings.
    Returns:
        Dataset: The dataset with the embeddings added as a new column 'embeddings'.
    """
    assert hasattr(model, 'encode'), 'model should have the method encode to get embeddings'
    dataset = dataset.map(
        lambda x: {'embeddings': get_embedding(model, x['fullText'], batch_size=batch_size, max_length=2048)},
        batched=True,
        batch_size=512,
        remove_columns=['fullText'],
        num_proc=None,
        desc="Extracting embeddings"
    )
    return dataset


def compute_metrics(queries: Dataset, candidate: Dataset):
    """
    Compute the metrics for the given queries and candidates using faiss
    Args:
        queries (Dataset): The queries dataset with the format {'authorIDs': str, 'documentID': str, 'embeddings': list}
        candidate (Dataset): The candidate dataset with the format {'authorIDs': str, 'documentID': str, 'embeddings': list}
    Returns:
        Dict: A dictionary with the metrics, including 'mrr', 'R@1', 'R@8'
    """
    # Convert the datasets to numpy arrays
    queries_embeddings = np.array(queries['embeddings']) # [n_queries, embedding_dim]
    candidate_embeddings = np.array(candidate['embeddings']) # [n_candidates, embedding_dim]
    candidate_authorIDs = candidate['authorIDs']
    queries_authorIDs = queries['authorIDs']
    # Verify all queries id are in the candidate
    if len(set(queries_authorIDs) - set(candidate_authorIDs)) > 0:
        print("Some queries authorIDs are not in the candidate authorIDs: ", set(queries_authorIDs) - set(candidate_authorIDs))
    author_dict = {authorID: i for i, authorID in enumerate(set(candidate_authorIDs))}
    # Convert authorIDs to indices
    candidate_authorIDs = np.array([author_dict[authorID] for authorID in candidate_authorIDs])
    queries_authorIDs = np.array([author_dict[authorID] for authorID in queries_authorIDs])

    # Create a faiss index
    index = faiss.IndexFlatL2(queries_embeddings.shape[1])
    index.add(candidate_embeddings)

    # Search for the nearest neighbors
    D, I = index.search(queries_embeddings, k=1000)

    # Compute the metrics
    MRR = 0.0
    R_1 = 0.0
    R_8 = 0.0
    for i in tqdm(range(len(queries)), desc="Computing metrics"):
        q_author_id = queries_authorIDs[i]
        # Find the rank of the correct document that have same authorID as the query
        rank = np.where(candidate_authorIDs[I[i]] == q_author_id)[0]
        if len(rank) > 0:
            rank = rank[0]
        else:
            rank = 1001 # No match found thus we assume the rank is 1001
        MRR += 1.0 / (rank + 1)
        R_1 += 1.0 if rank < 1 else 0.0
        R_8 += 1.0 if rank < 8 else 0.0
    MRR /= len(queries)
    R_1 /= len(queries)
    R_8 /= len(queries)

    return {
        'mrr': MRR,
        'R@1': R_1,
        'R@8': R_8
    }


def eval_hrs(
        model: Encoder,
        eval_batch_size: int,
        is_luar: bool = False,
        ):
    results = {}
    for domain in HRS_PATHS.keys():
        print(f"Evaluating {domain}")
        query_path = HRS_PATHS[domain]['resample_queries']
        candidate_path = HRS_PATHS[domain]['resample_candidates']
        ground_truth_path = HRS_PATHS[domain]['ground_truth']

        # Load the ground truth
        ground_truth = pd.read_json(ground_truth_path, lines=True)
        # Convert the ground truth to a dictionary with documentID as key and authorID as value
        docID_authorID_map = {}
        for i, row in ground_truth.iterrows():
            docID_authorID_map[row['documentID']] = row['authorIDs'][0]

        # Load the dataset
        queries = datasets.load_dataset('json', data_files=query_path, split='train')
        queries = queries.map(
            lambda x: {'authorIDs': docID_authorID_map[x['documentID']]},
            num_proc=8
        )
        collums_to_remove = set(queries.column_names) - {'authorIDs', 'documentID', 'fullText'}
        queries = queries.remove_columns(collums_to_remove)

        candidates = datasets.load_dataset('json', data_files=candidate_path, split='train')
        candidates = candidates.map(
            lambda x: {'authorIDs': docID_authorID_map[x['documentID']]},
            num_proc=8
        )
        collums_to_remove = set(candidates.column_names) - {'authorIDs', 'documentID', 'fullText'}
        candidates = candidates.remove_columns(collums_to_remove)

        # Extract the embeddings
        if is_luar:
            candidates = candidates.map(lambda x: {'fullText': [x['fullText']]}, num_proc=8)
            queries = queries.map(lambda x: {'fullText': [x['fullText']]}, num_proc=8)
            queries = extract_author_embeddings(model, queries)
            candidates = extract_author_embeddings(model, candidates)
        else:
            queries = extract_embeddings(model, queries, batch_size=eval_batch_size)
            candidates = extract_embeddings(model, candidates, batch_size=eval_batch_size)

        # normalize the embeddings 
        queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)

        # Compute the metrics
        metrics = compute_metrics(queries, candidates)
        results[f'{domain}/mrr'] = metrics['mrr']
        results[f'{domain}/R@1'] = metrics['R@1']
        results[f'{domain}/R@8'] = metrics['R@8']
        print(f"Results for {domain}: {metrics}")

    # Compute the average metrics
    avg_mrr = np.mean([results[f'{domain}/mrr'] for domain in HRS_PATHS.keys()])
    avg_r_1 = np.mean([results[f'{domain}/R@1'] for domain in HRS_PATHS.keys()])
    avg_r_8 = np.mean([results[f'{domain}/R@8'] for domain in HRS_PATHS.keys()])
    results.update({
        'avg/mrr': avg_mrr,
        'avg/R@1': avg_r_1,
        'avg/R@8': avg_r_8
    })
    print(f"Results for all domains: {results}")
    return results


def eval_amazon_reviews(model: Encoder, eval_batch_size: int, is_document_level: bool = False, is_luar: bool = False):
    query_path, candidate_path = AMAZON_REVIEWS_PATHS['queries'], AMAZON_REVIEWS_PATHS['candidates']
    queries = datasets.load_dataset('json', data_files=query_path, split='train')
    candidates = datasets.load_dataset('json', data_files=candidate_path, split='train')
    # Rename the columns to match the format
    queries = queries.rename_columns({'syms': 'fullText', 'author_id': 'authorIDs'})
    candidates = candidates.rename_columns({'syms': 'fullText', 'author_id': 'authorIDs'})
    # Flatten the dataset
    queries = queries.map(lambda x: flatten(x, 'authorIDs', 'fullText'), batched=True, remove_columns=queries.column_names)
    candidates = candidates.map(lambda x: flatten(x, 'authorIDs', 'fullText'), batched=True, remove_columns=candidates.column_names)
    # Preprocess the dataset
    queries = queries.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=8)
    candidates = candidates.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=8)
    if is_luar:
        # Group the dataset by authorIDs
        queries = group_by_column(queries, 'authorIDs', ['fullText'])
        candidates = group_by_column(candidates, 'authorIDs', ['fullText'])
        # Extract the embeddings
        queries = extract_author_embeddings(model, queries)
        candidates = extract_author_embeddings(model, candidates)
        # normalize the embeddings
        queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        # Compute the metrics
        metrics = compute_metrics(queries, candidates)
        print(f"Results for document-level Amazon reviews: {metrics}")
        return metrics
    else:
        queries = extract_embeddings(model, queries, batch_size=eval_batch_size)
        candidates = extract_embeddings(model, candidates, batch_size=eval_batch_size)
        if is_document_level:
            # normalize the embeddings 
            queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            # Compute the metrics
            metrics = compute_metrics(queries, candidates)
            print(f"Results for document-level Amazon reviews: {metrics}")
            return metrics
        else:
            # Group the dataset by authorIDs
            queries = group_by_column(queries, 'authorIDs', ['embeddings'])
            candidates = group_by_column(candidates, 'authorIDs', ['embeddings'])
            # Mean pool the embeddings
            queries = queries.map(lambda x: {'embeddings': np.mean(x['embeddings'], axis=0)}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': np.mean(x['embeddings'], axis=0)}, num_proc=8)
            # normalize the embeddings 
            queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            # Compute the metrics
            metrics = compute_metrics(queries, candidates)
            print(f"Results for author-level Amazon reviews: {metrics}")
            return metrics


def eval_MUD(model: Encoder, eval_batch_size: int, is_document_level: bool = False, is_luar: bool = False):
    query_path, candidate_path = MUD_PATHS['queries'], MUD_PATHS['candidates']
    queries = datasets.load_dataset('json', data_files=query_path, split='train')
    candidates = datasets.load_dataset('json', data_files=candidate_path, split='train')
    queries = queries.map(lambda x: {'fullText': x['syms'], 'authorIDs': x['author_id']}, remove_columns=queries.column_names)
    candidates = candidates.map(lambda x: {'fullText': x['syms'], 'authorIDs': x['author_id']}, remove_columns=candidates.column_names)
    # Flatten the dataset
    queries = queries.map(lambda x: flatten(x, 'authorIDs', 'fullText'), batched=True, remove_columns=queries.column_names)
    candidates = candidates.map(lambda x: flatten(x, 'authorIDs', 'fullText'), batched=True, remove_columns=candidates.column_names)
    # Preprocess the dataset
    queries = queries.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=8)
    candidates = candidates.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=8)
    if is_luar:
        # Group the dataset by authorIDs
        queries = group_by_column(queries, 'authorIDs', ['fullText'])
        candidates = group_by_column(candidates, 'authorIDs', ['fullText'])
        # Extract the embeddings
        queries = extract_author_embeddings(model, queries)
        candidates = extract_author_embeddings(model, candidates)
        # normalize the embeddings
        queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        # Compute the metrics
        metrics = compute_metrics(queries, candidates)
        print(f"Results for document-level Amazon reviews: {metrics}")
        return metrics
    else:
        queries = extract_embeddings(model, queries, batch_size=eval_batch_size)
        candidates = extract_embeddings(model, candidates, batch_size=eval_batch_size)
        if is_document_level:
            # normalize the embeddings 
            queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            # Compute the metrics
            metrics = compute_metrics(queries, candidates)
            print(f"Results for document-level MUD: {metrics}")
            return metrics
        else:
            # Group the dataset by authorIDs
            queries = group_by_column(queries, 'authorIDs', ['embeddings'])
            candidates = group_by_column(candidates, 'authorIDs', ['embeddings'])
            # Mean pool the embeddings
            queries = queries.map(lambda x: {'embeddings': np.mean(x['embeddings'], axis=0)}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': np.mean(x['embeddings'], axis=0)}, num_proc=8)
            # normalize the embeddings 
            queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            # Compute the metrics
            metrics = compute_metrics(queries, candidates)
            print(f"Results for author-level MUD: {metrics}")
            return metrics


def eval_PAN20(model: Encoder, eval_batch_size: int, is_document_level: bool = False, is_luar: bool = False):
    query_path, candidate_path = PAN20_PATHS['queries'], PAN20_PATHS['candidates']
    queries = datasets.load_dataset('json', data_files=query_path, split='train')
    candidates = datasets.load_dataset('json', data_files=candidate_path, split='train')
    queries = queries.map(lambda x: {'fullText': x['syms'], 'authorIDs': x['author_id']}, remove_columns=queries.column_names)
    candidates = candidates.map(lambda x: {'fullText': x['syms'], 'authorIDs': x['author_id']}, remove_columns=candidates.column_names)
    # Flatten the dataset
    queries = queries.map(lambda x: flatten(x, 'authorIDs', 'fullText'), batched=True, remove_columns=queries.column_names)
    candidates = candidates.map(lambda x: flatten(x, 'authorIDs', 'fullText'), batched=True, remove_columns=candidates.column_names)
    # Preprocess the dataset
    queries = queries.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=8)
    candidates = candidates.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=8)
    if is_luar:
        # Group the dataset by authorIDs
        queries = group_by_column(queries, 'authorIDs', ['fullText'])
        candidates = group_by_column(candidates, 'authorIDs', ['fullText'])
        # Extract the embeddings
        queries = extract_author_embeddings(model, queries)
        candidates = extract_author_embeddings(model, candidates)
        # normalize the embeddings
        queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        # Compute the metrics
        metrics = compute_metrics(queries, candidates)
        print(f"Results for document-level Amazon reviews: {metrics}")
        return metrics
    else:
        queries = extract_embeddings(model, queries, batch_size=eval_batch_size)
        candidates = extract_embeddings(model, candidates, batch_size=eval_batch_size)
        if is_document_level:
            # normalize the embeddings 
            queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            # Compute the metrics
            metrics = compute_metrics(queries, candidates)
            print(f"Results for document-level PAN20: {metrics}")
            return metrics
        else:
            # Group the dataset by authorIDs
            queries = group_by_column(queries, 'authorIDs', ['embeddings'])
            candidates = group_by_column(candidates, 'authorIDs', ['embeddings'])
            # Mean pool the embeddings
            queries = queries.map(lambda x: {'embeddings': np.mean(x['embeddings'], axis=0)}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': np.mean(x['embeddings'], axis=0)}, num_proc=8)
            # normalize the embeddings 
            queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            # Compute the metrics
            metrics = compute_metrics(queries, candidates)
            print(f"Results for author-level PAN20: {metrics}")
            return metrics
    

def eval_PAN21(model: Encoder, eval_batch_size: int, is_document_level: bool = False, is_luar: bool = False):
    query_path, candidate_path = PAN21_PATHS['queries'], PAN21_PATHS['candidates']
    queries = datasets.load_dataset('json', data_files=query_path, split='train')
    candidates = datasets.load_dataset('json', data_files=candidate_path, split='train')
    queries = queries.map(lambda x: {'fullText': x['syms'], 'authorIDs': x['author_id']}, remove_columns=queries.column_names)
    candidates = candidates.map(lambda x: {'fullText': x['syms'], 'authorIDs': x['author_id']}, remove_columns=candidates.column_names)
    # Flatten the dataset
    queries = queries.map(lambda x: flatten(x, 'authorIDs', 'fullText'), batched=True, remove_columns=queries.column_names)
    candidates = candidates.map(lambda x: flatten(x, 'authorIDs', 'fullText'), batched=True, remove_columns=candidates.column_names)
    # Preprocess the dataset
    queries = queries.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=8)
    candidates = candidates.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=8)
    if is_luar:
        # Group the dataset by authorIDs
        queries = group_by_column(queries, 'authorIDs', ['fullText'])
        candidates = group_by_column(candidates, 'authorIDs', ['fullText'])
        # Extract the embeddings
        queries = extract_author_embeddings(model, queries)
        candidates = extract_author_embeddings(model, candidates)
        # normalize the embeddings
        queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
        # Compute the metrics
        metrics = compute_metrics(queries, candidates)
        print(f"Results for document-level Amazon reviews: {metrics}")
        return metrics
    else:
        queries = extract_embeddings(model, queries, batch_size=eval_batch_size)
        candidates = extract_embeddings(model, candidates, batch_size=eval_batch_size)
        if is_document_level:
            # Compute the metrics
            metrics = compute_metrics(queries, candidates)
            print(f"Results for document-level PAN21: {metrics}")
            return metrics
        else:
            # Group the dataset by authorIDs
            queries = group_by_column(queries, 'authorIDs', ['embeddings'])
            candidates = group_by_column(candidates, 'authorIDs', ['embeddings'])
            # Mean pool the embeddings
            queries = queries.map(lambda x: {'embeddings': np.mean(x['embeddings'], axis=0)}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': np.mean(x['embeddings'], axis=0)}, num_proc=8)
            # normalize the embeddings 
            queries = queries.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            candidates = candidates.map(lambda x: {'embeddings': F.normalize(torch.tensor(x['embeddings']), p=2, dim=-1).numpy()}, num_proc=8)
            # Compute the metrics
            metrics = compute_metrics(queries, candidates)
            print(f"Results for author-level PAN21: {metrics}")
            return metrics


if __name__ == "__main__":
    import argparse
    from transformers import HfArgumentParser
    from src.model.qwen2_bidirectional import BidirectionalQwen2
    from src.args import (
        DataArguments, 
        ModelArguments, 
        TrainingArguments, 
        StyleEncoderDataArguments,
        DisentanglementDataArguments,
        EncoderModelArguments,
        DisentanglementModelArguments,
        EncoderTrainingArguments,
        DisentanglementTrainingArguments,
        )
    from src.model.model import AVAE
    
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_luar", action="store_true", help="Whether to use LUAR model for evaluation",
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to the yaml config file",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="Path to the model checkpoint",
    )

    args = parser.parse_args()
    if args.is_luar:
        wrapped_encoder = WrappedLUAR()
    else:
        config_file = args.config_file
        model_name_or_path = args.model_name_or_path
        if config_file is None:
            if model_name_or_path in ['Hieuman/qwen2-1.5b-hard-author-reps']:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
                model = BidirectionalQwen2.from_pretrained(model_name_or_path, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
            wrapped_encoder = WrappedEncoder(model, tokenizer, num_gpus=8)
        else:
            hf_parser = HfArgumentParser((DisentanglementDataArguments, DisentanglementModelArguments, DisentanglementTrainingArguments))
            data_args, model_args, training_args = hf_parser.parse_yaml_file(config_file)
            model = AVAE(
                        style_encoder_model_name_or_path=model_args.style_encoder_model_name_or_path,
                        content_encoder_model_name_or_path=model_args.content_encoder_model_name_or_path,
                        generator_model_name_or_path=model_args.generator_model_name_or_path,
                        embedding_dim=model_args.embedding_dim,
                        style_encoder_use_lora=model_args.style_encoder_use_lora,
                        content_encoder_use_lora=model_args.content_encoder_use_lora,
                        generator_use_lora=model_args.generator_use_lora,
                        lora_r=model_args.lora_r,
                        lora_alpha=model_args.lora_alpha,
                        lora_dropout=model_args.lora_dropout,
                        target_modules=model_args.target_modules,
                        attn_implementation=model_args.attn_implementation,
                        pooling_method=model_args.pooling_method,
                        dropout_prob=model_args.dropout_prob,
                        style_encoder_model_type=model_args.style_encoder_model_type,
                        content_encoder_model_type=model_args.content_encoder_model_type,
                        vae_loss_weight=model_args.vae_loss_weight,
                        reconstruction_loss_weight=model_args.reconstruction_loss_weight,
                        style_discriminator_loss_weight=model_args.style_discriminator_loss_weight,
                        content_discriminator_loss_weight=model_args.content_discriminator_loss_weight,
                        token_mi_reg_weight=model_args.token_mi_reg_weight,
                        mi_reg_weight=model_args.mi_reg_weight,
                        use_vae=model_args.use_vae,
                        style_loss_weight=model_args.style_loss_weight,
                        content_loss_weight=model_args.content_loss_weight,
                    )
            state_dict = torch.load(model_name_or_path, map_location='cpu')
            imcomplete_keys = model.load_state_dict(state_dict['model'], strict=False)
            print(f"Loaded model with missing keys: {imcomplete_keys}")
            # Style encoder evaluation
            wrapped_encoder = WrappedEncoder(model.style_encoder, num_gpus=8)
    
    # # Evaluate on HRS
    style_metrics = eval_hrs(model=wrapped_encoder,eval_batch_size=32, is_luar=args.is_luar)
    print(f"Style encoder evaluation results on HRS: {style_metrics}")
    # Evaluate on Amazon reviews
    amazon_reviews_metrics = eval_amazon_reviews(model=wrapped_encoder, eval_batch_size=32, is_document_level=False, is_luar=args.is_luar)
    print(f"Style encoder evaluation results on Amazon reviews: {amazon_reviews_metrics}")
    # Evaluate on MUD
    mud_metrics = eval_MUD(model=wrapped_encoder, eval_batch_size=32, is_document_level=False, is_luar=args.is_luar)
    print(f"Style encoder evaluation results on MUD: {mud_metrics}")
    # Evaluate on PAN20
    pan20_metrics = eval_PAN20(model=wrapped_encoder, eval_batch_size=32, is_document_level=False, is_luar=args.is_luar)
    print(f"Style encoder evaluation results on PAN20: {pan20_metrics}")
    # Evaluate on PAN21
    pan21_metrics = eval_PAN21(model=wrapped_encoder, eval_batch_size=32, is_document_level=False, is_luar=args.is_luar)
    print(f"Style encoder evaluation results on PAN21: {pan21_metrics}")