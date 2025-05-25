from dataclasses import asdict
import pathlib
import yaml
import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import partial
import datasets
import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import CPUOffload, ShardingStrategy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from lightning import seed_everything
from transformers import PreTrainedTokenizer, HfArgumentParser
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

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
from src.data_modules.style_dataloader import StyleDataModule
from src.data_modules.disentanglememt_dataloader import DisentanglementDataModule
from src.model.encoder import Encoder, WrappedEncoder
from src.model.generator import Generator
from src.model.model import AVAE
from src.model.utils import (
    choose_logger, 
    get_cosine_annealing_schedule_with_warmup, 
    get_trainable_parameters, 
    get_wrapping_policy, 
    get_activation_checkpointing_policy, 
    trainable_filter
)
from src.trainer.gradcache_trainer import GradCacheTrainer
from src.trainer.trainer import Trainer
from src.trainer.eval import eval_hrs

SHOULD_WRAP_MODULES = {Qwen2DecoderLayer, Qwen3DecoderLayer, Encoder, Generator}
SHOULD_WRAP_MODULES_NAMES = {Qwen2DecoderLayer.__name__, Qwen3DecoderLayer.__name__}


def get_dataloaders(
        fabric: L.Fabric,
        data_module: Union[StyleDataModule, DisentanglementDataModule],
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        epoch: int = 0,
        **kwargs: Any,
):  
    if isinstance(data_module, StyleDataModule):
        tokenizer = kwargs.get('tokenizer', None)
        if tokenizer is None:
            raise ValueError("Tokenizer is not provided")
        data_module.connect(
            model_type=model_args.model_type,
            tokenizer=tokenizer,
            world_size=fabric.world_size,
            global_rank=fabric.global_rank,
            global_batch_size=training_args.global_batch_size,
            max_seq_length=data_args.max_seq_length,
            num_train_example=data_args.number_of_training_samples,
            num_positives=data_args.num_positive,
            num_hard_negatives=data_args.num_hard_negative,
        )
        data_module.set_epoch(epoch)
        # use_dense_retrieval_hard_negatives = True if epoch > 0 else data_args.use_dense_retrieval_hard_negatives
        # style_encoder = kwargs.get('style_encoder', None)
        # num_clusters = data_args.num_clusters
        # threshold = data_args.threshold
        # do_filter = data_args.do_filter if epoch == 0 else False
        # if fabric.is_global_zero:
        #     data_module.prepare_data(
        #         use_dense_retrieval_hard_negatives=use_dense_retrieval_hard_negatives,
        #         style_encoder=style_encoder,
        #         num_clusters=num_clusters,
        #         threshold=threshold,
        #         do_filter=do_filter,
        #     )
        if fabric.is_global_zero:
            data_module.get_the_author_dict(model_checkpoint_dir=training_args.checkpoint_dir)
        fabric.barrier()
    elif isinstance(data_module, DisentanglementDataModule):
        style_encoder_tokenizer = kwargs.get('style_encoder_tokenizer', None)
        content_encoder_tokenizer = kwargs.get('content_encoder_tokenizer', None)
        generator_tokenizer = kwargs.get('generator_tokenizer', None)
        if style_encoder_tokenizer is None or generator_tokenizer is None or content_encoder_tokenizer is None:
            raise ValueError("Encoder and generator tokenizers are not provided")
        data_module.connect(
            style_encoder_tokenizer=style_encoder_tokenizer,
            content_encoder_tokenizer=content_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            max_length=data_args.max_seq_length,
            global_batch_size=training_args.global_batch_size,
            world_size=fabric.world_size,
            global_rank=fabric.global_rank,
            num_train_example=data_args.number_of_training_samples,
            prompt_loss=data_args.prompt_loss,
            placeholder_token=data_args.placeholder_token,
        )
        data_module.set_epoch(epoch)
    else:
        raise ValueError("Invalid data type")
    
    with fabric.rank_zero_first():
        data_module.setup(model_checkpoint_dir=training_args.checkpoint_dir)
        train_dataloader = data_module.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(train_dataloader, use_distributed_sampler=False, move_to_device=True)
    return train_dataloader


def main(
        fabric: L.Fabric,
        train_data: Union[StyleDataModule, DisentanglementDataModule],
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        ):
    fabric.seed_everything(training_args.seed)

    # Initialize model
    with fabric.rank_zero_first():
        if isinstance(model_args, EncoderModelArguments):
            model = Encoder(
                embedding_dim=model_args.embedding_dim,
                use_bidirectional=True,
                model_name_or_path=model_args.model_name_or_path,
                use_lora=model_args.use_lora,
                lora_r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                target_modules=model_args.target_modules,
                attn_implementation=model_args.attn_implementation,
                pooling_method=model_args.pooling_method,
                dropout_prob=model_args.dropout_prob,
                model_type=model_args.model_type,
                use_vae=model_args.use_vae,
            )
            tokenizer = model.tokenizer
            prepare_data_kwargs = {
                "tokenizer": tokenizer,
            }
        if isinstance(model_args, DisentanglementModelArguments):
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
            style_encoder_tokenizer = model.style_encoder.tokenizer
            content_encoder_tokenizer = model.content_encoder.tokenizer
            generator_tokenizer = model.generator.tokenizer
            prepare_data_kwargs = {
                "style_encoder_tokenizer": style_encoder_tokenizer,
                "content_encoder_tokenizer": content_encoder_tokenizer,
                "generator_tokenizer": generator_tokenizer,
            }
        else:
            raise ValueError("Invalid model type")
    
    fabric.barrier()
    trainable_params, all_param, trainable_params_percentage, trainable_layers = get_trainable_parameters(model)
    # Save whole model
    filter_fn = partial(trainable_filter, trainable_layers=trainable_layers) if trainable_params_percentage < 100 else None
    fabric.print(f"Number of trainable parameters: {trainable_params/1e6:.2f}M")
    fabric.print(f"Total number of parameters: {all_param/1e6:.2f}M")
    fabric.print(f"Percentage of trainable parameters: {trainable_params_percentage:.2f}%")
    model = fabric.setup_module(model)
    fabric.print("Model after wrapping")
    fabric.print(model)

    # Prepare dataloader
    train_dataloader = get_dataloaders(
        fabric=fabric,
        data_module=train_data,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        epoch=0,
        **prepare_data_kwargs,
    )
    fabric.barrier()

    # Setup the optimizer and scheduler
    step_per_epoch = len(train_dataloader) // training_args.num_accumulation_steps
    lr_max_steps = min(training_args.max_steps, step_per_epoch * training_args.max_epochs)
    num_epochs = lr_max_steps // step_per_epoch if lr_max_steps // step_per_epoch > 0 else 1
    warmup_steps = min(training_args.warmpup_proportion * step_per_epoch, 1000)
    warmup_steps = int(warmup_steps)
    lr = training_args.learning_rate
    min_lr = training_args.min_learning_rate
    min_reduce_rate = min_lr / lr
    fabric.print(f"Number of training examples: {len(train_dataloader.dataset)}")
    fabric.print(f"Effective batch size: {training_args.effective_batch_size}") # i.e. num_accumulation_steps * global_batch_size
    fabric.print(f"Global batch size: {training_args.global_batch_size}") # i.e. num_nodes * num_gpus * batch_size_per_gpu
    fabric.print(f"Number of gradient accumulation steps: {training_args.num_accumulation_steps}")
    fabric.print(f"Number of steps per epoch: {step_per_epoch}")
    fabric.print(f"Number of warmup steps: {warmup_steps}")
    fabric.print(f"Number of max steps: {lr_max_steps}")
    fabric.print(f"Number of epochs: {num_epochs}")
    fabric.print(f"Initial learning rate: {lr}")
    fabric.print(f"Minimum learning rate: {min_lr}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_cosine_annealing_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=lr_max_steps,
        num_cycles=num_epochs,
        min_reduce_rate=min_reduce_rate,
    )

    # Load the checkpoint if needed
    state = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": 0,
        "epoch_num": 0,
    }
    if training_args.checkpoint_file is not None:
        checkpoint_path = training_args.checkpoint_file
        if os.path.exists(checkpoint_path):
            fabric.print(f"Load checkpoint from {checkpoint_path}")
            if training_args.only_load_model:
                model_state = {'model': model}
                fabric.load(checkpoint_path, model_state, strict=False)
                model = model_state.pop("model")
            else:
                state['model'] = model
                fabric.load(checkpoint_path, state, strict=False)
                model = state.pop("model")
    
    if isinstance(training_args, EncoderTrainingArguments):
        train_encoder(
            fabric=fabric,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            data_module=train_data,
            state=state,
            model=model,
            train_dataloader=train_dataloader,
            lr_max_steps=lr_max_steps,
            filter_fn=filter_fn,
        )
    elif isinstance(training_args, DisentanglementTrainingArguments):
        train_vae(
            fabric=fabric,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            data_module=train_data,
            state=state,
            model=model,
            train_dataloader=train_dataloader,
            lr_max_steps=lr_max_steps,
            filter_fn=filter_fn,
            num_epochs=num_epochs,
        )


def train_encoder(
        fabric: L.Fabric,
        data_args: StyleEncoderDataArguments,
        model_args: EncoderModelArguments, 
        training_args: EncoderTrainingArguments,
        data_module: StyleDataModule,
        state: Dict[str, Any],
        model: Encoder,
        train_dataloader: torch.utils.data.DataLoader,
        lr_max_steps: int,
        filter_fn: Optional[callable] = None,
        ):
    trainer = GradCacheTrainer(
        fabric=fabric,
        chunk_size=training_args.gc_chunk_size,
        loss_type=training_args.loss_type,
        temperature=training_args.temperature,
        use_miner=training_args.use_miner,
        margin=training_args.margin,
    )

    checkpoint_epoch = state["epoch_num"]
    iter_num = state["iter_num"]
    for epoch in range(checkpoint_epoch, training_args.max_epochs):
        fabric.print("Start training")
        if epoch != 0: # Prepare the dataloader for the next epoch, except the first epoch because it is already prepared
            # data_module.preprocess_data_dir = os.path.join(training_args.checkpoint_dir, 'preprocess_data')
            # if fabric.is_global_zero and iter_num == 0: # Only preprocess data in the first iteration of each epoch
            #     fabric.print("Preparing dataloader for the next epoch")
            #     model_prams = model.hprams
            #     eval_model = Encoder(**model_prams)
            #     # check if cuda is available
            #     if torch.cuda.is_available():
            #         device = torch.device('cuda')
            #     else:
            #         device = torch.device('cpu')
            #     eval_model.to(device)
            #     if torch.cuda.device_count() > 1:
            #         eval_model = torch.nn.DataParallel(eval_model)
            #     data_module.prepare_data(
            #         use_dense_retrieval_hard_negatives=True,
            #         style_encoder=eval_model,
            #         num_clusters=data_args.num_clusters,
            #         threshold=data_args.threshold,
            #         do_filter=False,
            #     )
            #     del eval_model
            #     torch.cuda.empty_cache()
            # fabric.barrier()
            train_dataloader = get_dataloaders(
                fabric=fabric,
                data_module=data_module,
                data_args=data_args,
                model_args=model_args,
                training_args=training_args,
                tokenizer=model.tokenizer,
                epoch=epoch,
            )
            
        fabric.barrier()
        checkpoint_path = trainer.fit_epoch(
            model=model,
            train_dataloader=train_dataloader,
            state=state,
            lr_max_steps=lr_max_steps,
            grad_norm_clip=training_args.grad_norm_clip,
            log_interval=training_args.log_interval,
            checkpoint_iterval=training_args.checkpoint_interval,
            checkpoint_dir=training_args.checkpoint_dir,
            checkpoint_filter=filter_fn,
            eval_batch_size=training_args.eval_batch_size,
        )
        torch.cuda.empty_cache()
        state['model'] = model
        fabric.load(checkpoint_path, state, strict=False)
        model = state.pop("model")
        fabric.barrier()
    fabric.print("Training finished")


def train_vae(
        fabric: L.Fabric,
        data_args: StyleEncoderDataArguments,
        model_args: EncoderModelArguments, 
        training_args: EncoderTrainingArguments,
        data_module: StyleDataModule,
        num_epochs: int,
        state: Dict[str, Any],
        model: AVAE,
        train_dataloader: torch.utils.data.DataLoader,
        lr_max_steps: int,
        filter_fn: Optional[callable] = None,
        ):

    trainer = Trainer(
        fabric=fabric,
        lr_max_steps=lr_max_steps,
        num_accumulation_steps=training_args.num_accumulation_steps,
        grad_norm_clip=training_args.grad_norm_clip,
        log_interval=training_args.log_interval,
        checkpoint_iterval=training_args.checkpoint_interval,
        checkpoint_dir=training_args.checkpoint_dir,
        checkpoint_filter=filter_fn,
        eval_batch_size=training_args.eval_batch_size,
        num_kl_weight_cycles=1, 
    )

    checkpoint_epoch = state["epoch_num"]
    for epoch in range(checkpoint_epoch, num_epochs):
        fabric.print("Start training with epoch: ", epoch)
        if epoch != 0: # Prepare the dataloader for the next epoch, except the first epoch because it is already prepared
            prepare_data_kwargs = {
                "style_encoder_tokenizer": model.style_encoder.tokenizer,
                "content_encoder_tokenizer": model.content_encoder.tokenizer,
                "generator_tokenizer": model.generator.tokenizer,
            }
            train_dataloader = get_dataloaders(
                fabric=fabric,
                data_module=data_module,
                data_args=data_args,
                model_args=model_args,
                training_args=training_args,
                epoch=epoch,
                **prepare_data_kwargs,
            )
        fabric.barrier()
        checkpoint_path = trainer.fit_epoch(
            model=model,
            train_dataloader=train_dataloader,
            state=state,
        )
        torch.cuda.empty_cache()
        state['model'] = model
        fabric.load(checkpoint_path, state, strict=False)
        model = state.pop("model")
        fabric.barrier()
    fabric.print("Training finished")


def setup(
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        run_name: str = None,
        ):
    seed_everything(training_args.seed)

    if isinstance(data_args, StyleEncoderDataArguments):
        assert isinstance(model_args, EncoderModelArguments), "EncoderModelArguments should be used with StyleEncoderDataArguments"
        assert isinstance(training_args, EncoderTrainingArguments), "EncoderTrainingArguments should be used with StyleEncoderDataArguments"
        train_data = StyleDataModule(
            seed=training_args.seed,
            num_workers=training_args.num_workers,
            preprocess_data_dir=data_args.preprocess_data_dir,
        )
        ALL_CUSTOM_LAYERS = None
    elif isinstance(data_args, DisentanglementDataArguments):
        assert isinstance(model_args, DisentanglementModelArguments), "DisentanglementModelArguments should be used with DisentanglementDataArguments"
        assert isinstance(training_args, DisentanglementTrainingArguments), "DisentanglementTrainingArguments should be used with DisentanglementDataArguments"
        train_data = DisentanglementDataModule(
            seed=training_args.seed,
            num_workers=training_args.num_workers
        )
        from src.model.vae import VQVAE, VAE
        ALL_CUSTOM_LAYERS = (VQVAE, VAE)
    else:
        raise ValueError("Invalid data type")
    
    strategy = training_args.strategy
    if training_args.nodes > 1 or training_args.devices > 1:
        if training_args.strategy == 'fsdp':
            # Config sharding strategy
            if training_args.sharding_strategy == "full_shard":
                sharding_strategy = ShardingStrategy.FULL_SHARD
            elif training_args.sharding_strategy == "shard_grad_op":
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            elif training_args.sharding_strategy == "ddp":
                sharding_strategy = ShardingStrategy.NO_SHARD
            elif training_args.sharding_strategy == "hybrid_full_shard":
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            elif training_args.sharding_strategy == "hybrid_shard_grad_op":
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
            else:
                raise ValueError("Invalid sharding strategy")
            wrapping_policy = get_wrapping_policy(SHOULD_WRAP_MODULES, custom_layer_cls=ALL_CUSTOM_LAYERS, class_names=SHOULD_WRAP_MODULES_NAMES)
            activation_checkpointing_policy = get_activation_checkpointing_policy(SHOULD_WRAP_MODULES, custom_layer_cls=ALL_CUSTOM_LAYERS, class_names=SHOULD_WRAP_MODULES_NAMES) 
            
            strategy = FSDPStrategy(
                auto_wrap_policy=wrapping_policy,
                activation_checkpointing_policy=activation_checkpointing_policy if training_args.activation_checkpointing else None,
                sharding_strategy=sharding_strategy,
                limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
                state_dict_type="full",
                cpu_offload=training_args.use_cpu_offload,
                timeout=datetime.timedelta(days=2), # makeing large timeout for model reindexing in each epoch
            )
        elif training_args.strategy == 'ddp':
            strategy = DDPStrategy(
                find_unused_parameters=True, 
                timeout=datetime.timedelta(days=2) # making large timeout for model reindexing in each epoch
                ) 
    else:
        strategy = "auto"

    logger_dir = os.path.join(training_args.checkpoint_dir, f"logs_{training_args.logger_type}")
    os.makedirs(logger_dir, exist_ok=True)
    logger = choose_logger(
        logger_name=training_args.logger_type,
        out_dir=Path(logger_dir),
        run_name=run_name,
        project_name=training_args.logger_name,
        log_interval=training_args.log_interval,
    )

    # check whether gpu is support bf16 if not set precision to 32
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        training_args.precision = '32-true'
    
    fabric = L.Fabric(
        accelerator='gpu',
        strategy=strategy,
        devices=training_args.devices,
        num_nodes=training_args.nodes,
        precision=training_args.precision,
        loggers=logger,
    )

    fabric.launch(
        main,
        train_data=train_data,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
    )


if __name__ == "__main__":
    import argparse
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_encoder", action="store_true", help="Train encoder"
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the yaml config file",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Run name"
    )
    parser.add_argument(
        "--nodes", type=int, default=None, help="Number of nodes"
    )
    parser.add_argument(
        "--devices", type=int, default=None, help="Number of devices"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--checkpoint_file", type=str, default=None, help="Checkpoint file to resume training"
    )
    parser.add_argument(
        "--only_load_model", action="store_true", help="Only load the model from the checkpoint"
    )
    # Finetuning args
    parser.add_argument(
        "--global_batch_size", type=int, default=None, help="Global batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--content_discriminator_loss_weight", type=float, default=1.0, help="Content discriminator loss weight"
    )
    parser.add_argument(
        "--style_discriminator_loss_weight", type=float, default=1.0, help="Style discriminator loss weight"
    )

    args = parser.parse_args()
    config_file = args.config_file

    if args.train_encoder:
        hf_parser = HfArgumentParser((StyleEncoderDataArguments, EncoderModelArguments, EncoderTrainingArguments))
    else:
        hf_parser = HfArgumentParser((DisentanglementDataArguments, DisentanglementModelArguments, DisentanglementTrainingArguments))
    data_args, model_args, training_args = hf_parser.parse_yaml_file(config_file)

    # Update the arguments with the command line arguments
    if args.global_batch_size is not None:
        training_args.global_batch_size = args.global_batch_size
    if args.learning_rate is not None:
        training_args.learning_rate = args.learning_rate
    if args.checkpoint_dir is not None:
        training_args.checkpoint_dir = args.checkpoint_dir
    if args.checkpoint_file is not None:
        training_args.checkpoint_file = args.checkpoint_file
    if args.nodes is not None:
        training_args.nodes = args.nodes
    if args.devices is not None:
        training_args.devices = args.devices
    if args.run_name is not None:
        training_args.run_name = args.run_name
    if args.content_discriminator_loss_weight is not None:
        model_args.content_discriminator_loss_weight = args.content_discriminator_loss_weight
    if args.style_discriminator_loss_weight is not None:
        model_args.style_discriminator_loss_weight = args.style_discriminator_loss_weight
    
    training_args.compute_batch_sizes()
    # Save the configuration file
    config_path = Path(training_args.checkpoint_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(asdict(data_args), f)
        yaml.dump(asdict(model_args), f)
        yaml.dump(asdict(training_args), f)

    setup(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        run_name=args.run_name,
    )


