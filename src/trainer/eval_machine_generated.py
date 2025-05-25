# Modified from https://github.com/LLNL/LUAR/tree/main/fewshot_iclr2024

from collections import defaultdict
import json
import logging
import os
import random
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    pairwise_distances,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from ERLAS.model.erlas import ERLAS, WrappedERLAS

from src.model.encoder import WrappedEncoder


def random_combinations(lists, k=10):
    """Returns k random combinations of len(lists) where we randomly draw
       one element from each list.
    """
    i = 0
    seen = {}
    result = []
    while i < k:
        sample = []
        for l in lists: 
            sample.append(random.choice(l))

        key = "_".join([str(s) for s in sample])
        if key in seen: 
            continue

        seen[key] = True
        result.append(sample)
        i += 1
    return result

def index_generator(
    INDEX_TO_AUTHOR, 
    machine_support_labels,
    machine_queries_labels,
    support_to_queries,
    max_trials,
    mode
):
    """Generate the support and query indices for the given model.
    """
    if mode == "multiple_target":
        lm_indices = [np.where(machine_support_labels == lm_index)[0].tolist() for lm_index, name in INDEX_TO_AUTHOR.items() if name not in ["opt", "gpt2"]]
        lm_indices_comb = random_combinations(lm_indices, k=max_trials)
        
        for trial, support_indices in enumerate(lm_indices_comb):
            to_delete = [support_to_queries[support_index] for support_index in support_indices]
            to_delete = [item for sublist in to_delete for item in sublist]
            queries_index = np.delete(
                np.arange(len(machine_queries_labels)),
                to_delete,
            )
            yield "ALL", trial, support_indices, queries_index
    else:
        for lm_index, LM in INDEX_TO_AUTHOR.items():
            if LM in ["opt", "gpt2"]: continue
            trial = 0
            
            for support_index in np.where(machine_support_labels == lm_index)[0]:

                try:
                    queries_index = np.where(machine_queries_labels == lm_index)[0]
                    to_delete = np.argwhere(queries_index == support_to_queries[support_index])
                    queries_index = np.delete(
                        queries_index,
                        to_delete
                    )
                except:
                    # this shouldn't hurt too much
                    queries_index = np.where(machine_queries_labels == lm_index)[0]

                yield LM, trial, support_index, queries_index
                
                trial += 1
                if trial >= max_trials:
                    break

def calculate_roc_metrics(
        labels, 
        scores,
    ):
    """Calculates the Detection metrics for the given labels and scores.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    roc_auc = roc_auc_score(labels, scores)
    roc_auc_cutoff = roc_auc_score(labels, scores, max_fpr=10**-2)
    
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc": roc_auc,
        "roc_auc_cutoff": roc_auc_cutoff,
    }

def calculate_nn_metrics(
        INDEX_TO_AUTHOR,
        machine_support_embeddings, machine_support_labels,
        machine_queries_embeddings, machine_queries_labels,
        background_embeddings,
        support_to_queries,
        max_trials,
        mode,
    ):
    """Calculates the LUAR metrics for the given embeddings.
    """
    background_distances = pairwise_distances(
        machine_support_embeddings, 
        Y=background_embeddings,
        metric="cosine", 
        n_jobs=-1
    )

    metrics = defaultdict(dict)
    metrics["global"] = defaultdict(float)
    all_roc_metrics = []
    generator = index_generator(
        INDEX_TO_AUTHOR, 
        machine_support_labels, 
        machine_queries_labels, 
        support_to_queries, 
        max_trials,
        mode,
    )
    
    for index in generator:
        LM, trial, support_index, query_indices = index

        if mode == "multiple_target" or mode == "multiple_target_paraphrase":
            if mode == "multiple_target_paraphrase":
                offset = machine_support_embeddings.shape[0] // 2
                support_index = [support_index, support_index + offset]
            
            distances = []
            for sindex in support_index:
                dist_background = background_distances[sindex]
                dist_target = pairwise_distances(
                    machine_support_embeddings[sindex].reshape(1, -1),
                    Y=machine_queries_embeddings[query_indices],
                    metric="cosine",
                ).flatten()
                dist = np.concatenate((dist_target, dist_background), axis=0)
                distances.append(dist)
            dist = np.vstack(distances).min(axis=0)
        else:
            dist_background = background_distances[support_index]
            dist_target = pairwise_distances(
                machine_support_embeddings[support_index].reshape(1, -1), 
                Y=machine_queries_embeddings[query_indices], 
                metric="cosine",
            ).flatten()
            dist = np.concatenate((dist_target, dist_background), axis=0)

        labels = [1 for _ in range(len(dist_target))] + [0 for _ in range(len(dist_background))]
        roc_metrics = calculate_roc_metrics(labels, (-dist).tolist())
        metrics[LM]["trial_{:03d}".format(trial)] = roc_metrics
        all_roc_metrics.append(roc_metrics)

    metrics["global"]["roc"] = all_roc_metrics
    return metrics

def build_support_to_queries_map(
    machine_support_labels, machine_queries_labels,
    support_size, queries_size
):
    """Builds a mapping from the support indices to the queries indices.
    """
    if support_size % queries_size != 0:
        logging.warning("support_size % queries_size != 0, this may cause overlap between the supports and queries sometimes.")
    if support_size < queries_size:
        raise ValueError("support_size < queries_size, this is not supported!")

    counts = np.unique(machine_queries_labels, return_counts=True)[1]
    queries_LM_start = np.insert(counts, 0, 0).cumsum().tolist()
    
    support_to_queries = {}

    last_LM_label, LM_index = 0, 0
    chunksize = support_size // queries_size
    for support_index, support_label in enumerate(machine_support_labels):
        if support_label != last_LM_label:
            last_LM_label = support_label
            LM_index = 0

        offset = LM_index * chunksize
        start = queries_LM_start[support_label] + offset
        end = start + chunksize
        # don't violate LM boundaries, or go over the number of queries
        end = min(end, queries_LM_start[support_label+1])
        
        support_to_queries[support_index] = np.arange(start, end).tolist()
        LM_index += 1

    return support_to_queries
    
def create_index_to_author(machine_episodes):
    """Creates a mapping from the index to the author.
    """
    unique = pd.unique(machine_episodes[["author_id", "author"]].values.ravel())
    keys = unique[::2]
    values = unique[1::2]
    INDEX_TO_AUTHOR = dict(zip(keys, values))
    logging.info(f"INDEX_TO_AUTHOR={INDEX_TO_AUTHOR}")
    return INDEX_TO_AUTHOR

def extract_embeddings(episodes:List[List[str]], model: WrappedEncoder, num_tokens=2048, batch_size=128):
    """Extract embeddings from the model for a list of episodes.
    """
    embeddings = []
    for i in tqdm(range(0, len(episodes), batch_size)):
        batch = episodes[i:i+batch_size]
        B, E = len(batch), len(batch[0])

        batch = [j for i in batch for j in i]    
        output = model.encode(texts=batch, max_length=num_tokens, batch_size=batch_size) # [bs * num_docs, dim]
        output = output.reshape(B, E, -1).mean(dim=1) # [bs, dim]
        output = F.normalize(output, p=2, dim=-1) # [bs, dim]
        embeddings.append(output.detach().cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)


def extract_author_embeddings(episodes:List[List[str]], model: WrappedEncoder, num_tokens=512, batch_size=128):
    """Extract embeddings from the model for a list of episodes.
    """
    embeddings = []
    for i in tqdm(range(0, len(episodes), batch_size)):
        batch = episodes[i:i+batch_size]
        output = model.encode(texts=batch, max_length=num_tokens, batch_size=batch_size) # [bs, dim]
        output = F.normalize(output, p=2, dim=-1) # [bs, dim]
        embeddings.append(output.detach().cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)


def evaluate(
        dataset_dirname, 
        model: WrappedEncoder, 
        mode: str,
        paraphrase_p: int=0, 
        support_size: int=5, 
        eval_batch_size: int=128,
        max_trials: int=10000000,
        is_author_level: bool=False,
        ):
    print(f"Evaluating {dataset_dirname}")
    dataset_name = os.path.basename(dataset_dirname)
    nrows = None
    nbackground = 3000
    if paraphrase_p > 0.:
        me_path = os.path.join(dataset_dirname, f"machine_episodes_num-tokens=128_p={paraphrase_p}_L=20_episode-size={support_size}_support.jsonl")
        machine_support = pd.read_json(me_path, lines=True, nrows=nrows)
        machine_support_labels = np.array(machine_support.author_id.tolist())
        me_path = os.path.join(dataset_dirname, f"machine_episodes_num-tokens=128_p={paraphrase_p}_L=20_episode-size={support_size}_queries.jsonl")
        machine_queries = pd.read_json(me_path, lines=True, nrows=nrows)
        machine_queries_labels = np.array(machine_queries.author_id.tolist())
    else:
        me_path = os.path.join(dataset_dirname, f"machine_episodes_num-tokens=128_episode-size={support_size}.jsonl")
        machine_episodes = pd.read_json(me_path, lines=True, nrows=nrows)
        machine_support = machine_episodes.copy()
        machine_support_labels = np.array(machine_support.author_id.tolist())
        machine_queries = machine_episodes.copy()
        machine_queries_labels = np.array(machine_queries.author_id.tolist())
    background = pd.read_json(os.path.join(dataset_dirname, f"background_num-tokens=128_episode-size={support_size}_num-background=3000.jsonl"), lines=True, nrows=nbackground)
    logging.info(f"background size = {len(background)}")
    
    INDEX_TO_AUTHOR = create_index_to_author(machine_queries)
    
    support_to_queries = build_support_to_queries_map(
        machine_support_labels, machine_queries_labels,
        support_size, support_size
    )
    extract_embeddings_fn = extract_embeddings if not is_author_level else extract_author_embeddings

    machine_support_embeddings = extract_embeddings_fn(machine_support.syms.tolist(), model, batch_size=eval_batch_size)
    if args.mode == "multiple_target_paraphrase":
        machine_support_paraphrase_embeddings = extract_embeddings_fn(machine_support.syms_paraphrase.tolist(), model, batch_size=eval_batch_size)
        machine_support_embeddings = np.concatenate([machine_support_embeddings, machine_support_paraphrase_embeddings], axis=0)
    machine_queries_embeddings = extract_embeddings_fn(machine_queries.syms.tolist(), model, batch_size=eval_batch_size)
    background_embeddings = extract_embeddings_fn(background.syms.tolist(), model, batch_size=eval_batch_size)

    print("Calculate the metrics")
    metrics = calculate_nn_metrics(
            INDEX_TO_AUTHOR,
            machine_support_embeddings, machine_support_labels,
            machine_queries_embeddings, machine_queries_labels,
            background_embeddings,
            support_to_queries,
            max_trials,
            mode,
        )
    
    return metrics



if __name__ == "__main__":
    import argparse
    from transformers import HfArgumentParser, AutoTokenizer, AutoModel
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
    from src.model.qwen2_bidirectional import BidirectionalQwen2
    
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, help="Path to the yaml config file",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--is_author_level", action="store_true",
        help="Whether the model is trained on author level episodes.",
    )
    parser.add_argument("--mode", type=str, default="single_target",
                    choices=["single_target", "multiple_target", "multiple_target_paraphrase"])
    parser.add_argument("--paraphrase_p", type=float, default=0.)
    parser.add_argument("--max_trials", type=int, default=999999999,
                    help="Number of trials to run for each machine query & target.")

    args = parser.parse_args()
    config_file = args.config_file
    model_name_or_path = args.model_name_or_path
    if config_file is None and args.is_author_level == False:
        model_str = model_name_or_path
        if model_name_or_path in ['Hieuman/qwen2-1.5b-hard-author-reps']:
                model_str = 'Qwen2Bidirectional'
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
                model = BidirectionalQwen2.from_pretrained(model_name_or_path, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        wrapped_encoder = WrappedEncoder(model, tokenizer, num_gpus=8)
    elif args.is_author_level:
        model_str = 'ERLAS'
        model = ERLAS.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        wrapped_encoder = WrappedERLAS(model, tokenizer, num_gpus=8)
    else:
        model_str = 'AVAE'
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

    data_dir = "/home/hieum/uonlp/LUAR/fewshot_iclr2024/data/fewshot_data"
    support_sizes = [1, 5, 10]
    for dirname in os.listdir(data_dir):
        dataset_dirname = os.path.join(data_dir, dirname)
        paraphrase_str = f"_p={args.paraphrase_p}_L=20" if args.paraphrase_p > 0. else ""
        print(f"Evaluating {dataset_dirname}")
        for support_size in support_sizes:
            metrics_fname = os.path.join(
                dataset_dirname, 
                f"metrics_{model_str}_mode={args.mode}{paraphrase_str}_{support_size}_{support_size}_128.json"
            )
            metrics = evaluate(
                dataset_dirname, 
                wrapped_encoder, 
                mode=args.mode,
                paraphrase_p=args.paraphrase_p, 
                support_size=support_size, 
                eval_batch_size=32,
                max_trials=args.max_trials,
                is_author_level=args.is_author_level,
            )
            with open(metrics_fname, "w") as fout:
                json.dump(metrics, fout, indent=4)