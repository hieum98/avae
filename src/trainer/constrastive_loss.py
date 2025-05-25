from typing import Optional, Tuple
import einops
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F 
from pytorch_metric_learning import losses, distances

from src.trainer.surrogate_miner import SurrogateMiner
from src.trainer.utils import mismatched_sizes_all_gather

class ContrastiveLoss:
    def __init__(
            self,
            loss_type: str = 'NTXentLoss', # 'NTXentLoss' or 'SupConLoss'
            temperature: float = 0.05,
            use_miner: bool = False,
            margin: float = 0.1, # margin for surrogate miner to filter out false negatives
            ) -> None:
        self.temperature = temperature if temperature > 0 else 1.0
        distance = distances.CosineSimilarity()
        if loss_type == 'NTXentLoss':
            self.loss_fn = losses.NTXentLoss(temperature=temperature, distance=distance)
        elif loss_type == 'SupConLoss':
            self.loss_fn = losses.SupConLoss(temperature=temperature, distance=distance)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        if use_miner:
            self.miner = SurrogateMiner(margin=margin, distance=distance, collect_stats=True)
    
    def __call__(
            self,
            reps: torch.Tensor, # [batch_size, emb_dim]
            labels: torch.Tensor = None, # [batch_size]
            indices_tuple: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, # Tuple of one-hot tensors for anchors, positives, and negatives  
    ):  
        if torch.distributed.is_initialized():
            reps = mismatched_sizes_all_gather(reps)
            reps = torch.cat(reps, dim=0) # [world_size * batch_size, emb_dim]
            if labels is not None:
                labels = mismatched_sizes_all_gather(labels)
                labels = torch.cat(labels, dim=0) # [world_size * batch_size]
            if indices_tuple is not None:
                anchors, positives, negatives = indices_tuple
                anchors = mismatched_sizes_all_gather(anchors)
                anchors = torch.cat(anchors, dim=0)
                positives = mismatched_sizes_all_gather(positives)
                positives = torch.cat(positives, dim=0)
                negatives = mismatched_sizes_all_gather(negatives)
                negatives = torch.cat(negatives, dim=0)
                indices_tuple = (anchors, positives, negatives)
        
        miner_stats = {}
        if hasattr(self, 'miner') and indices_tuple is None:
            # use surrogate miner to generate indices_tuple
            indices_tuple = self.miner(reps, labels)
            miner_stats = {
                'num_pos_pairs': self.miner.num_pos_pairs,
                "num_neg_pairs": self.miner.num_neg_pairs,
                "num_triplets": self.miner.num_triplets,
                "avg_pos_pair_dist": self.miner.pos_pair_dist,
                "avg_neg_pair_dist": self.miner.neg_pair_dist,
                "avg_triplet_margin": self.miner.avg_triplet_margin,
            }

        loss = self.loss_fn(reps, labels, indices_tuple) # Loss over global batch
        return loss, miner_stats
        
            



