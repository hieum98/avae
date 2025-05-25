import functools
import math
from pathlib import Path
from typing import Any, Literal, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from torch.optim.lr_scheduler import LambdaLR
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from transformers import PreTrainedModel, Conv1D
from peft.tuners.lora import LoraLayer


def find_all_linear_names(model: nn.Module, quantization: Optional[bool] = False):
    if not isinstance(model, PreTrainedModel):
        raise ValueError("Model must be an instance of `transformers.PreTrainedModel`")
    
    if quantization:
        from bitsandbytes.nn import Linear4bit

        cls = (Linear4bit, Conv1D)
    else:
        cls = (torch.nn.Linear, Conv1D)

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.rsplit(".", 1)[-1]  # get the base name
            lora_module_names.add(names)
            
    if "lm_head" in lora_module_names:  
        lora_module_names.remove("lm_head")

    # ignore the last classification head for text generation models
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
        lora_module_names -= {last_module_name}
        
    return list(lora_module_names)


def get_wrapping_policy(transformer_layers, custom_layer_cls: Tuple=None, class_names: List[str]=None):
    """
    A wrapping policy for AuthorRepsModel that wraps:
    1. all leaf modules with requires_grad=True.
    2. all sequential modules with all children have requires_grad=True.
    3. all LoraLayer.
    4. all transformer layers with a specific transformer_layer_cls.
    """
    def lambda_policy_fn(module, custom_layer_cls=None, class_names=None):
        # All leaf modules with requires_grad=True
        is_trainable_layer = (len(list(module.named_children())) == 0) and (getattr(module, "weight", None) is not None) and (module.weight.requires_grad)
        is_trainable_seqential = isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module if hasattr(m, "weight"))
        is_lora_layer = isinstance(module, LoraLayer)
        # Check if the module is a custom layer
        if custom_layer_cls is not None:
            is_custom_layer = isinstance(module, custom_layer_cls)
        else:
            is_custom_layer = False
        if class_names is not None:
            module_name = module.__class__.__name__
            module_should_be_wrapped = True if module_name in class_names else False 
        return is_trainable_layer or is_trainable_seqential or is_lora_layer or is_custom_layer or module_should_be_wrapped
    lambda_policy_fn = functools.partial(lambda_policy_fn, custom_layer_cls=custom_layer_cls, class_names=class_names)
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layers,
    ) 
    policies=[lambda_policy, transformer_wrap_policy]
    return functools.partial(_or_policy, policies=policies)


def get_activation_checkpointing_policy(transformer_layers, custom_layer_cls: Tuple=None, class_names: List[str]=None):
    """
    A wrapping activation checkpointing policy for AuthorRepsModel that wraps:
    1. all sequential modules with requires_grad=True. We assume all additional modules in the sequential module
    2. all transformer layers with a specific transformer_layer_cls.
    """
    def lambda_policy_fn(module, custom_layer_cls, class_names=None):
        # Check if the module is a Sequential and all the children have requires_grad=True
        if custom_layer_cls is not None:
            is_custom_layer = isinstance(module, custom_layer_cls)
        else:
            is_custom_layer = False
        if class_names is not None:
            module_name = module.__class__.__name__
            module_should_be_wrapped = True if module_name in class_names else False 
        return (isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module if hasattr(m, "weight"))) or is_custom_layer or module_should_be_wrapped
    lambda_policy_fn = functools.partial(lambda_policy_fn, custom_layer_cls=custom_layer_cls, class_names=class_names)
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layers,
    )
        
    policies=[lambda_policy, transformer_wrap_policy]
    return functools.partial(_or_policy, policies=policies)


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb"],
    run_name: str,
    out_dir: Path,
    project_name: str,
    log_interval: int = 1,
    resume: Optional[bool] = None,
    **kwargs: Any,
    ):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name=run_name, flush_logs_every_n_steps=log_interval, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name=run_name, **kwargs)
    if logger_name == "wandb":
        return WandbLogger(name=run_name, project=project_name, resume=resume, **kwargs)
    raise ValueError(f"`--logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb'.")


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model (`PreTrainedModel`):
            The model to print the number of trainable parameters for.

    Returns:
        `Tuple[int, int, float]`:
            The number of trainable parameters, the total number of parameters and the
            percentage of trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    trainable_layers = []
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            trainable_layers.append(name)

    return trainable_params, all_param, 100 * trainable_params / all_param, trainable_layers


def get_cycle_sigmoid_vae_kl_weight(
        current_step: int,
        num_training_steps: int,
        num_cycles: float = 1.0,
    ) -> float:
    """
    Get the cycle sigmoid KL weight for VAE training.

    Args:
        current_step (int): Current training step.
        num_training_steps (int): Total number of training steps.
        num_cycles (float): Number of cycles for the KL weight schedule.

    Returns:
        float: The KL weight for the current step.
    """
    num_training_steps_per_cycle = num_training_steps // num_cycles
    if current_step >= num_training_steps:
        return 1.0
    # Get current step in the current epoch
    current_step = current_step % num_training_steps_per_cycle
    # Sigmoid schedule learning rate from 0.0 to 1.0
    progress = (current_step) / max(1, num_training_steps_per_cycle)
    sigmoid_kl_weight = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))
    return sigmoid_kl_weight


def get_cosine_annealing_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 1,
        min_reduce_rate: float = 0.0,
        last_epoch: int = -1,
    ) -> LambdaLR:

    def lr_lambda(current_step):
        num_training_steps_per_cycle = num_training_steps // num_cycles
        if current_step >= num_training_steps:
            return min_reduce_rate
        # Get current step in the current epoch
        current_step = current_step % num_training_steps_per_cycle
        # Linearly increase learning rate from min_reduce_rate to 1.0 over num_warmup_steps
        if current_step < num_warmup_steps:
            return  min_reduce_rate + (1.0 - min_reduce_rate) * current_step / max(1, num_warmup_steps)
        # Cosin schedule learning rate from 1.0 to min_reduce_rate
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps_per_cycle - num_warmup_steps
        )
        cosine_lr_multiple = 0.5 * (
            1.0 + min_reduce_rate + math.cos(math.pi * progress) * (1.0 - min_reduce_rate)
        )
        return max(min_reduce_rate, cosine_lr_multiple)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def trainable_filter(key: str, value: Any, trainable_layers: List[str]=[]) -> bool:
    if any([layer in key for layer in trainable_layers]):
        print("Layer to save: ", key)
        return True
    else:
        return False


