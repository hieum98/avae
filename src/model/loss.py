import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, distances


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        torch.distributed.all_gather(
            tensor_list, tensor, group=group, async_op=async_op
        )
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True)
            for i in range(torch.distributed.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None


all_gather_with_grad = AllGather.apply

def mismatched_sizes_all_gather(
    tensor: torch.Tensor, group=None, async_op=False, mismatched_axis=0
):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert torch.distributed.is_initialized(), "torch.distributed not initialized"
    world_size = torch.distributed.get_world_size()
    # let's get the sizes for everyone
    mismatched_sizes = torch.tensor(
        [tensor.shape[mismatched_axis]], dtype=torch.int64, device="cuda"
    )
    sizes = [torch.zeros_like(mismatched_sizes) for _ in range(world_size)]
    torch.distributed.all_gather(
        sizes, mismatched_sizes, group=group, async_op=async_op
    )
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    padded = torch.zeros(
        (
            *tensor.shape[:mismatched_axis],
            max_size,
            *tensor.shape[mismatched_axis + 1 :],
        ),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    # selects the place where we're adding information
    padded_to_fill = padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis])
    padded_to_fill[...] = tensor
    # gather the padded tensors
    tensor_list = [
        torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype)
        for _ in range(world_size)
    ]
    all_gather_with_grad(tensor_list, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        # checks that the rest is 0
        assert (
            not tensor_list[rank]
            .narrow(
                mismatched_axis,
                sizes[rank],
                padded.shape[mismatched_axis] - sizes[rank],
            )
            .count_nonzero()
            .is_nonzero()
        ), "This would remove non-padding information"
        tensor_list[rank] = tensor_list[rank].narrow(mismatched_axis, 0, sizes[rank])
    return tensor_list


def pairwise_angle_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the absolute normalized angle distance. See :class:`~sentence_transformers.losses.AnglELoss`
    or https://arxiv.org/abs/2309.12871v1 for more information.

    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.

    Returns:
        Tensor: Vector with res[i] = angle_sim(a[i], b[i])
    """

    # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
    # chunk both tensors to obtain complex components
    a, b = torch.chunk(x, 2, dim=1)
    c, d = torch.chunk(y, 2, dim=1)

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
    re /= dz / dw
    im /= dz / dw

    norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
    return torch.abs(norm_angle)

def cosine_loss(
        txt1_emb: torch.Tensor, # [batch_size, hidden_dim]
        txt2_emb: torch.Tensor, 
        labels: torch.Tensor, # [batch_size]
        tau: float = 20.0
        ) -> torch.Tensor:
    """
    Compute cosine loss
    Args:
        txt1_emb: [batch_size, hidden_dim] The first text embedding
        txt2_emb: [batch_size, hidden_dim] The second text embedding
        labels: [batch_size] The labels of the text pairs (i.e., 1 for similar, 0 for dissimilar)
        tau: temperature
    Returns:
        loss: [1]

    """  # NOQA
    # modified from: https://github.com/bojone/CoSENT/blob/124c368efc8a4b179469be99cb6e62e1f2949d39/cosent.py#L79
    # label matrix indicating which pairs are similar
    labels = (labels[:, None] < labels[None, :]).float()
    
    # compute cosine similarity
    txt1_emb = F.normalize(txt1_emb, p=2, dim=1)
    txt2_emb = F.normalize(txt2_emb, p=2, dim=1)
    scores = torch.sum(txt1_emb * txt2_emb, dim=1) * tau # [batch_size]
    scores = scores[:, None] - scores[None, :] # [batch_size, batch_size]

    # mask out the dissimilar pairs
    scores = (scores - (1 - labels) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(scores.device)
    scores = torch.concat((zero, scores), dim=0)
    return torch.logsumexp(scores, dim=0)


def angle_loss(
        txt1_emb: torch.Tensor, # [batch_size, hidden_dim]
        txt2_emb: torch.Tensor, 
        labels: torch.Tensor, # [batch_size]
        tau: float = 20.0,
        ):
    # modified from: https://arxiv.org/pdf/2309.12871v9
    # label matrix indicating which pairs are similar
    labels = (labels[:, None] < labels[None, :]).float()

    # compute angle similarity
    scores = pairwise_angle_sim(txt1_emb, txt2_emb) * tau
    scores = scores[:, None] - scores[None, :]

    # mask out the dissimilar pairs
    scores = (scores - (1 - labels) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(scores.device)
    scores = torch.concat((zero, scores), dim=0)
    return torch.logsumexp(scores, dim=0)


class AngleLoss:
    def __init__(
            self,
            cosine_w: float = 0.0,
            ibn_w: float = 1.0,
            angle_w: float = 0.02,
            cosine_tau: float = 20.0,
            ibn_tau: float = 20.0,
            angle_tau: float = 20.0,
    ):
        self.cosine_w = cosine_w
        self.ibn_w = ibn_w
        self.angle_w = angle_w
        self.cosine_tau = cosine_tau
        self.angle_tau = angle_tau
        distance = distances.CosineSimilarity()
        temperature = 1.0 / ibn_tau
        self.ibn_loss = losses.SupConLoss(temperature=temperature, distance=distance)
    
    def __call__(self, 
            txt1_emb: torch.Tensor, # [batch_size, hidden_dim]
            txt2_emb: torch.Tensor, 
            labels: torch.Tensor, # [batch_size]
            ) -> torch.Tensor:
        """
        Compute the loss
        Args:
            txt1_emb: [batch_size, hidden_dim] The first text embedding
            txt2_emb: [batch_size, hidden_dim] The second text embedding
            labels: [batch_size] The labels of the text pairs (i.e., 1 for similar, 0 for dissimilar)
        Returns:
            loss: [1]
        """
        is_distributed = torch.distributed.is_initialized()
        if is_distributed:
            world_size = torch.distributed.get_world_size()
            txt1_emb = mismatched_sizes_all_gather(txt1_emb)
            txt2_emb = mismatched_sizes_all_gather(txt2_emb)
            labels = mismatched_sizes_all_gather(labels)
            txt1_emb = torch.cat(txt1_emb, dim=0) # [world_size * batch_size, hidden_dim]
            txt2_emb = torch.cat(txt2_emb, dim=0)
            labels = torch.cat(labels, dim=0) # [world_size * batch_size]
        else:
            world_size = 1
        
        # if all labels are 0 return 0
        if torch.sum(labels) == 0:
            return 0.0

        loss = 0.0
        if self.cosine_w > 0:
            loss += self.cosine_w * cosine_loss(txt1_emb, txt2_emb, labels, tau=self.cosine_tau) * world_size
        if self.angle_w > 0:
            loss += self.angle_w * angle_loss(txt1_emb, txt2_emb, labels, tau=self.angle_tau) * world_size
        if self.ibn_w > 0:
            bs = labels.size(0)
            txt1_labels = torch.arange(bs, device=labels.device)
            txt2_labels = txt1_labels + bs
            for i in range(bs):
                if labels[i] == 1:
                    # label for ith sample in txt1 and txt2 is the same if they have label 1 (i.e., ith instance in labels)
                    txt2_labels[i] = txt1_labels[i]
            labels = torch.cat([txt1_labels, txt2_labels], dim=0)
            embs = torch.cat([txt1_emb, txt2_emb], dim=0)
            loss = loss + self.ibn_w * self.ibn_loss(embs, labels) * world_size
        return loss
        