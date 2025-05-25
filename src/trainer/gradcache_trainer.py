from collections import UserDict
import pathlib
from contextlib import nullcontext
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import typing
from einops import repeat
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import get_device_states, set_device_states
import lightning as L

from src.model.encoder import Encoder
from src.trainer.constrastive_loss import ContrastiveLoss


def split_input(model_input, chunk_size: int) -> List:
    """
    Split model input into chunks.
    :param model_input: model input
    :param chunk_size: chunk size
    :return: list of input chunks with same format as model_input
    """
    if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, torch.Tensor) for x in model_input.values()):
        keys = list(model_input.keys())
        chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
        return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    elif isinstance(model_input, list) and all(isinstance(x, torch.Tensor) for x in model_input):
        chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
        return [list(s) for s in zip(*chunked_x)]

    elif isinstance(model_input, torch.Tensor):
        return list(model_input.split(chunk_size, dim=0))

    elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
        args_chunks = split_input(model_input[0], chunk_size)
        kwargs_chunks = split_input(model_input[1], chunk_size)
        return list(zip(args_chunks, kwargs_chunks))
    
    elif isinstance(model_input, tuple) and list(map(type, model_input)) == [dict, dict]:
        args_chunks = split_input(model_input[0], chunk_size) # list of dicts
        global_kwargs = model_input[1]
        for args_chunk in args_chunks:
            args_chunk.update(global_kwargs)
        return args_chunks
    
    else:
        raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


class GradCacheTrainer:
    def __init__(
            self,
            fabric: L.Fabric,
            chunk_size: Optional[int] = 1,
            loss_type: str = 'NTXentLoss', # 'NTXentLoss' or 'SupConLoss'
            temperature: float = 0.05,
            use_miner: bool = False,
            margin: float = 0.1, # margin for surrogate miner to filter out false negatives
            ) -> None:
        self.fabric = fabric
        self.chunk_size = chunk_size

        self.loss = ContrastiveLoss(
            loss_type=loss_type,
            temperature=temperature,
            use_miner=use_miner,
            margin=margin,
        )

        self.best_result = None

    def get_input_tensors(self, model_input) -> List[torch.Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, torch.Tensor):
            return [model_input]
        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])
        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])
        else:
            return []
    
    def forward_no_grad(
            self, 
            model: Encoder, 
            model_inputs: Dict[str, torch.Tensor],
            ):
        with torch.no_grad():
            rnd_state = RandContext(*self.get_input_tensors(model_inputs))
            input_ids = model_inputs['input_ids'] # (batch_size, sample_size, seq_len)
            attn_mask = model_inputs['attention_mask'] # (batch_size, sample_size, seq_len)
            prompt_length = model_inputs.get('prompt_length', None) # (batch_size, sample_size)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                prompt_length=prompt_length,
            )
            reps = outputs['reps'] # (batch_size, emb_dim)
        
        return reps, rnd_state
    
    @typing.no_type_check
    def build_cache(
            self,
            reps: torch.Tensor, # [batch_size, emb_dim]
            labels: torch.Tensor = None, # [batch_size]
            indices_tuple: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, # Tuple of one-hot tensors for anchors, positives, and negatives  
            ):
        reps = reps.detach().requires_grad_(True)
        with nullcontext():
            with self.fabric.autocast():
                loss, miner_stats = self.loss(reps, labels, indices_tuple=indices_tuple)
        self.fabric.backward(loss)
        cache = reps.grad

        loss = loss.detach()
        return cache, loss, miner_stats
    
    def forward_backward(
            self,
            model: Encoder,
            model_inputs: Dict[str, torch.Tensor],
            state: RandContext,
            cache: torch.Tensor, # [batch_size, emb_dim]
            ):
        with state:
            input_ids = model_inputs['input_ids']
            attn_mask = model_inputs['attention_mask']
            prompt_length = model_inputs.get('prompt_length', None) # (batch_size, sample_size)

            # Forward pass
            outputs = model(input_ids, attn_mask, prompt_length)
            reps = outputs['reps'] # [batch_size, emb_dim]

            # Backward pass
            surrogate_loss = torch.dot(reps.flatten(), cache.flatten())
            self.fabric.backward(surrogate_loss)

    def train_step(
            self,
            model: Encoder,
            batch: Dict[str, torch.Tensor],
            ) -> Dict[str, Any]:
        lables = batch.pop('labels', None) # [batch_size]
        indices_tuple = batch.pop('indices_tuple', None) # Tuple of one-hot tensors for anchors, positives, and negatives
        splitted_inputs = split_input(batch, self.chunk_size)

        # Forward pass for each chunk
        rnd_state = []
        all_reps = []
        for model_inputs in splitted_inputs:
            # model_inputs is [chunk_size, ...]
            reps, state = self.forward_no_grad(model, model_inputs)
            rnd_state.append(state)
            all_reps.append(reps)
        all_reps = torch.cat(all_reps, dim=0) # [batch_size, emb_dim]

        # Build cache
        cache, loss, miner_stats = self.build_cache(all_reps, labels=lables, indices_tuple=indices_tuple)
        self.fabric.barrier() # wait for all processes to finish building cache
        cache = cache.split(self.chunk_size, dim=0) 

        # Forward-backward pass for each chunk
        accumulated_flags = [True for _ in range(len(splitted_inputs)-1)] + [False]
        for model_inputs, state, rep_cache, flag in zip(splitted_inputs, rnd_state, cache, accumulated_flags):
            with self.fabric.no_backward_sync(model, enabled=flag):
                self.forward_backward(model, model_inputs, state, rep_cache)
        
        output = {'loss': loss,}
        if miner_stats:
            output.update(miner_stats)
        return output
    
    def fit_epoch(
            self,
            model: Encoder,
            train_dataloader: DataLoader,
            state: Dict[str, Any],
            lr_max_steps: int = 1000,
            grad_norm_clip: float = None,
            log_interval: int = 1,
            checkpoint_iterval: Optional[int] = 10000,
            checkpoint_dir: Optional[str] = './checkpoints/',
            checkpoint_filter: Optional[Callable] = None,
            eval_batch_size: Optional[int] = 32,
            ):
        # Get the state of training
        optimizer: torch.optim.Optimizer = state["optimizer"]
        scheduler : torch.optim.lr_scheduler.LambdaLR = state.get("scheduler", None)
        checkpoint_iter_num = state.get("iter_num", 0) # checkpoint iteration number inner epoch
        epoch_num = state.get("epoch_num", 0) # checkpoint epoch number
        self.fabric.print(f"Starting epoch {epoch_num} with {len(train_dataloader)} iterations")

        # Train the model
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            # Restore the state of training by going to the saved datapoint
            if batch_idx < checkpoint_iter_num:
                continue
            total_steps = (epoch_num * len(train_dataloader)) + batch_idx
            if total_steps > lr_max_steps:
                break
            if epoch_num == 0 and batch_idx == 0:
                size_info = {k: v.size() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                self.fabric.print("First batch data: {}".format(size_info))
            
            # Train step
            iter_t0 = time.perf_counter()
            output = self.train_step(model, batch)
            if grad_norm_clip is not None:
                self.fabric.clip_gradients(model, optimizer, max_norm=grad_norm_clip)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            # Log the training step
            if batch_idx % log_interval == 0:
                t1 = time.perf_counter()
                metrics = {
                    'iter_time': t1 - iter_t0,
                    'epoch': epoch_num,
                    'batch_idx': batch_idx,
                    'lr': scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'],
                }
                metrics.update(output)
                self.fabric.log_dict(metrics, step=total_steps)
                self.fabric.print(f"Epoch {epoch_num} Batch {batch_idx}: loss {output['loss']:.4f}, iter_time {metrics['iter_time']:.4f}, lr {metrics['lr']:.4f}")

            # Save checkpoint and evaluate
            checkpoint_path = None
            if batch_idx != 0 and (batch_idx % checkpoint_iterval == 0 or batch_idx == len(train_dataloader)-1):
                checkpoint_path = pathlib.Path(checkpoint_dir) / f"lastest.ckpt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "iter_num": batch_idx + 1 if batch_idx < len(train_dataloader)-1 else 0,
                    "epoch_num": epoch_num if batch_idx < len(train_dataloader)-1 else epoch_num + 1,
                }
                if checkpoint_filter is not None:
                    self.fabric.save(checkpoint_path, state, filter={'model': checkpoint_filter})
                else:
                    self.fabric.save(checkpoint_path, state)
                self.fabric.print(f"Checkpoint saved at {checkpoint_path}")
                self.fabric.barrier()
        return checkpoint_path

        

