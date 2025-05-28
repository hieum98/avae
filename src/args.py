from dataclasses import dataclass, field
import os
from typing import List, Optional, Union


@dataclass
class DataArguments:
    """Data arguments for the training process."""
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for the input text."}
    )
    number_of_training_samples: int = field(
        default=100000,
        metadata={"help": "Number of training samples accounting for all datasets, i.e., the sum of all datasets."}
    )
    preprocess_data_dir: str = field(
        default=None,
        metadata={"help": "Directory to save the preprocessed data."}
    )

@dataclass
class ModelArguments:
    """Model arguments related to the model architecture."""
    model_name_or_path: str = field(
        metadata={"help": "Model name or path of the pretrained model"}
    )
    model_type: str = field(
        metadata={"help": "Model type, e.g., 'llama3', 'qwen2', etc. This is used to determine the template to use."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA or not"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R parameter"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for LoRA"}
    )
    target_modules: Union[str, List[str]] = field(
        default="all",
        metadata={"help": "Target modules for LoRA"}
    )
    attn_implementation: str = field(
        default='sdpa',
        metadata={"help": "Attention implementation to use. Can be eager/sdpa/flash_attention_2"}
    )

@dataclass
class TrainingArguments:
    """
    Arguments related to training
    """
    seed: int = field(
        default=777,
        metadata={"help": "Seed for reproducibility"}
    )
    nodes: int = field(
        default=1,
        metadata={"help": "Number of nodes to use for training"}
    )
    devices: int = field(
        default=1,
        metadata={"help": "Number of devices per node to use for training"}
    )
    precision: str = field(
        default='bf16-true',
        metadata={"help": "Precision to use. Can be bf16-true/bf16-mixed/16-mixed/32"}
    )
    strategy: str = field(
        default='fsdp',
        metadata={"help": "Strategy to use. Currently only supports dpp and fsdp"}
    )
    use_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to use CPU offload or not"}
    )
    sharding_strategy: str = field(
        default='full_shard',
        metadata={"help": "Sharding strategy to use. Can be full_shard/shard_grad_op/ddp/hybrid_full_shard/hybrid_shard_grad_op"}
    )
    activation_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use activation checkpointing or not"}
    )
    global_batch_size: int = field(
        default=32,
        metadata={"help": "Global batch size"}
    )
    eval_batch_size: int = field(
        default=32,
        metadata={"help": "Evaluation batch size"}
    )
    num_workers: int = field(
        default=1,
        metadata={"help": "Number of workers to use for data loading"}
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Maximum number of epochs to train"}
    )
    max_steps: int = field(
        default=float("inf"),
        metadata={"help": "Maximum number of steps to train"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"}
    )
    min_learning_rate: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate"}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to apply."},
        )
    warmpup_proportion: float = field(
        default=0.1,
        metadata={"help": "Proportion of training steps to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."}
    )
    grad_norm_clip: float = field(
        default=1.0,
        metadata={"help": "Gradient norm clipping value"}
    )
    checkpoint_dir: str = field(
        default=None,
        metadata={"help": "Directory to save checkpoints"}
    )
    checkpoint_file: str = field(
        default=None,
        metadata={"help": "File to save checkpoints"}
    )
    only_load_model: bool = field(
        default=False,
        metadata={"help": "Whether to only load the model or not"}
    )
    checkpoint_interval: int = field(
        default=1000,
        metadata={"help": "Interval to save the checkpoint"}
    )
    logger_type: str = field(
        default='wandb',
        metadata={"help": "Name of the logger to use. Can be wandb/tensorboard"}
    )
    logger_name: str = field(
        default='default',
        metadata={"help": "Name of the logger"}
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "Interval to log the training progress"}
    )


@dataclass
class StyleEncoderDataArguments(DataArguments):
    """Data arguments for the style encoder."""
    num_positive: int = field(
        default=1,
        metadata={"help": "Number of positive samples per document"}
    )
    num_hard_negative: int = field(
        default=64,
        metadata={"help": "Number of hard negative samples per document"}
    )
    num_clusters: int = field(
        default=256,
        metadata={"help": "Number of clusters to use for clustering"}
    )
    threshold: float = field(
        default=0.5,
        metadata={"help": "Threshold for filtering easy documents"}
    )
    do_filter: bool = field(
        default=False,
        metadata={"help": "Whether to filter easy documents or not"}
    )
    use_dense_retrieval_hard_negatives: bool = field(
        default=None,
        metadata={"help": "Whether to use dense retrieval for hard negatives, if None do nothing"}
    )


@dataclass
class DisentanglementDataArguments(DataArguments):
    """Data arguments for the disentanglement model."""
    prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to compute the loss on the prompt tokens or not"}
    )
    placeholder_token: str = field(
        default=" <|placeholder|> ",
        metadata={"help": "Placeholder token to use for the generator"}
    )


@dataclass
class EncoderModelArguments(ModelArguments):
    """Model arguments for the encoder."""
    pooling_method: str = field(
        default='mean',
        metadata={"help": "Pooling method to use. Can be mean/cls"}
    )
    dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability"}
    )
    embedding_dim: int = field(
        default=1536,
        metadata={"help": "Embedding dimension of the model for both content and style encoders"}
    )
    # use_bidirectional: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to use bidirectional encoder or not"}
    # )
    use_vae: bool = False # Always False for encoder training
    


@dataclass
class DisentanglementModelArguments(ModelArguments):
    """Model arguments for the disentanglement model."""
    style_encoder_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Model name or path of the pretrained style encoder model"}
    )
    content_encoder_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Model name or path of the pretrained content encoder model"}
    )
    generator_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Model name or path of the pretrained generator model"}
    )
    embedding_dim: int = field(
        default=1536,
        metadata={"help": "Embedding dimension of the model for both content and style encoders"}
    )
    style_encoder_use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for the style encoder or not"}
    )
    content_encoder_use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for the content encoder or not"}
    )
    generator_use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for the generator or not"}
    )
    pooling_method: str = field(
        default='mean',
        metadata={"help": "Pooling method to use. Can be mean/cls"}
    )
    dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability"}
    )
    style_encoder_model_type: str = field(
        default='qwen2',
        metadata={"help": "Model type for the encoder. Can be llama3/qwen2/llama2"}
    )
    content_encoder_model_type: str = field(
        default='qwen2',
        metadata={"help": "Model type for the encoder. Can be llama3/qwen2/llama2"}
    )
    vae_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the VAE loss"}
    )
    reconstruction_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the reconstruction loss"}
    )
    style_discriminator_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the style discriminator loss"}
    )
    content_discriminator_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for the content discriminator loss"}
    )
    token_mi_reg_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the token mutual information regularization loss"}
    )
    mi_reg_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the mutual information regularization loss"}
    )
    use_vae: bool = field(
        default=False,
        metadata={"help": "Whether to use VAE or not"}
    )
    style_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the style KLD loss"}
    )
    content_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the content KLD loss"}
    )
    constraint_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the constraint loss, i.e., the loss that ensures the model does not move far from the preference model"}
    )


@dataclass
class EncoderTrainingArguments(TrainingArguments):
    """Training arguments for the encoder."""
    gc_chunk_size: int = field(
        default=1,
        metadata={"help": "GradCache chunk size. If None, not use GradCache."}
    )
    loss_type: str = field(
        default='SupConLoss',
        metadata={"help": "Loss type to use. Can be NTXentLoss/SupConLoss"}
    )
    temperature: float = field(
        default=0.02,
        metadata={"help": "Temperature parameter for the loss function"}
    )
    use_miner: bool = field(
        default=True,
        metadata={"help": "Whether to use miner or not"}
    )
    margin: float = field(
        default=0.1,
        metadata={"help": "Margin parameter for filtering the false negatives"}
    )
    def compute_batch_sizes(self):
        # Ensure that the global batch size is a multiple of the number of devices
        if self.global_batch_size % (self.devices * self.nodes) != 0:
            print(f"Warning: global_batch_size {self.global_batch_size} is not a multiple of devices {self.devices} * nodes {self.nodes}.")
            self.global_batch_size = self.devices * self.nodes * (self.global_batch_size // (self.devices * self.nodes) + 1)
            print(f"Setting global_batch_size to {self.global_batch_size} to be a multiple of devices * nodes.")
            self.effective_batch_size = self.global_batch_size
            self.num_accumulation_steps = 1
        else:
            self.effective_batch_size = self.global_batch_size
            self.num_accumulation_steps = 1


@dataclass
class DisentanglementTrainingArguments(TrainingArguments):
    """Training arguments for the disentanglement model."""
    effective_batch_size: int = field(
        default=32,
        metadata={"help": "Effective batch size"}
    )

    def compute_batch_sizes(self):
        # Ensure that the global batch size is a multiple of the number of devices
        if self.global_batch_size % (self.nodes * self.devices) != 0:
            print(f"Warning: global_batch_size {self.global_batch_size} is not a multiple of nodes * devices {self.nodes * self.devices}.")
            self.global_batch_size = (self.global_batch_size // (self.nodes * self.devices) + 1) * (self.nodes * self.devices)
            print(f"Setting global_batch_size to {self.global_batch_size} to be a multiple of nodes * devices.")
        # Ensure that the effective batch size is a multiple of the global batch size
        if self.effective_batch_size % self.global_batch_size != 0:
            print(f"Warning: effective_batch_size {self.effective_batch_size} is not a multiple of global_batch_size {self.global_batch_size}.")
            self.effective_batch_size = self.global_batch_size * (self.effective_batch_size // self.global_batch_size + 1)
            print(f"Setting effective_batch_size to {self.effective_batch_size} to be a multiple of global_batch_size.")
            self.num_accumulation_steps = self.effective_batch_size // self.global_batch_size
        else:
            self.num_accumulation_steps = self.effective_batch_size // self.global_batch_size
    
    



    
