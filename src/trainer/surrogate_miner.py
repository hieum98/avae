import torch

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.miners.base_miner import BaseMiner


class SurrogateMiner(BaseMiner):
    """
    Returns triplets (a, p, n) that sim(a, p) - sim(a, n) > margin or dis(a, n) - dis(a, p) > margin.
    This is to filter out the triplets maybe false negatives
    Args:
        margin
    """

    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )

        self.set_stats(ap_dist, an_dist, triplet_margin)

        threshold_condition = triplet_margin > self.margin
        
        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            with torch.no_grad():
                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()
