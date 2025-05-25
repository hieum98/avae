import pathlib
import shutil
import time
from typing import Any, Callable, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from transformers import BatchEncoding

from src.model.model import AVAE
from src.model.utils import get_cycle_sigmoid_vae_kl_weight
from src.model.encoder import Encoder, WrappedEncoder
from src.trainer.eval import eval_hrs


class Trainer():
    def __init__(
            self,
            fabric: L.Fabric,
            num_kl_weight_cycles: int = 1,
            lr_max_steps: int = 1000,
            num_accumulation_steps: int = 1,
            grad_norm_clip: float = None,
            log_interval: int = 1,
            checkpoint_iterval: Optional[int] = 10000,
            checkpoint_dir: Optional[str] = './checkpoints/',
            checkpoint_filter: Optional[Callable] = None,
            eval_batch_size: Optional[int] = 32,
            ):
        self.fabric = fabric
        self.best_results = {}
        self.num_kl_weight_cycles = num_kl_weight_cycles
        self.lr_max_steps = lr_max_steps
        self.num_accumulation_steps = num_accumulation_steps
        self.grad_norm_clip = grad_norm_clip
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_iterval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filter = checkpoint_filter
        self.eval_batch_size = eval_batch_size

    def fit_epoch(
            self,
            model: AVAE,
            train_dataloader: DataLoader,
            state: Dict[str, Any],
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
            if total_steps > self.lr_max_steps * self.num_accumulation_steps:
                break
            if epoch_num == 0 and batch_idx == 0:
                size_info = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        size_info[k] = v.size()
                    elif isinstance(v, BatchEncoding):
                        size_info[k] = {
                            'input_ids': v['input_ids'].size(),
                            'attention_mask': v['attention_mask'].size(),
                            'labels': v['labels'].size() if 'labels' in v else None,
                        }
                self.fabric.print("First batch data: {}".format(size_info))
            
            # Train step
            iter_t0 = time.perf_counter()
            kl_weight = get_cycle_sigmoid_vae_kl_weight(
                current_step=total_steps,
                num_training_steps=self.lr_max_steps * self.num_accumulation_steps,
                num_cycles=self.num_kl_weight_cycles,
            )
            is_accumulation = batch_idx % self.num_accumulation_steps != 0 
            # Do not accumulate gradients for checkpointing batches and last batch
            if batch_idx % self.checkpoint_interval == 0 or batch_idx + 1 == len(train_dataloader):
                is_accumulation = False
            with self.fabric.no_backward_sync(model, enabled=is_accumulation):
                model_input = {
                    'style_encoder_inputs_ids': batch['style_encoder_input_tokenized']['input_ids'],
                    'style_encoder_attention_mask': batch['style_encoder_input_tokenized']['attention_mask'],
                    'content_encoder_inputs_ids': batch['content_encoder_input_tokenized']['input_ids'],
                    'content_encoder_attention_mask': batch['content_encoder_input_tokenized']['attention_mask'],
                    'reconstruct_txt_inputs_ids': batch['txt_reconstruct_tokenized']['input_ids'],
                    'reconstruct_txt_attention_mask': batch['txt_reconstruct_tokenized']['attention_mask'],
                    'reconstruct_labels': batch['txt_reconstruct_tokenized']['labels'],
                    'txt_placeholder_token_pos': batch['txt_placeholder_token_pos'],
                    'style_discriminator_input_ids': batch['style_discriminator_input_tokenized']['input_ids'],
                    'style_discriminator_attention_mask': batch['style_discriminator_input_tokenized']['attention_mask'],
                    'style_discriminator_labels': batch['style_discriminator_input_tokenized']['labels'],
                    'style_placeholder_token_pos': batch['style_placeholder_token_pos'],
                    'content_discriminator_input_ids': batch['content_discriminator_input_tokenized']['input_ids'],
                    'content_discriminator_attention_mask': batch['content_discriminator_input_tokenized']['attention_mask'],
                    'content_discriminator_labels': batch['content_discriminator_input_tokenized']['labels'],
                    'content_placeholder_token_pos': batch['content_placeholder_token_pos'],
                    'kl_loss_weight': kl_weight,
                    'style_labels': batch['style_labels'],
                    'content_labels': batch['content_labels'],
                }
                model_output = model(**model_input)
                loss = model_output['loss']
                # Scale the loss for gradient accumulation
                self.fabric.backward(loss / self.num_accumulation_steps)
            
            if not is_accumulation:
                if self.grad_norm_clip is not None:
                    self.fabric.clip_gradients(model, optimizer, max_norm=self.grad_norm_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            # Log metrics
            # Detach the model output from the graph to avoid memory leaks
            model_output = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_output.items()
            }
            if batch_idx % self.log_interval == 0:
                t1 = time.perf_counter()
                metrics = {
                    'iter_time': t1 - iter_t0,
                    'epoch': epoch_num,
                    # 'iter_num': batch_idx,
                    'kl_weight': kl_weight,
                    'lr': scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'],
                }
                metrics.update(model_output)
                self.fabric.log_dict(metrics, step=total_steps)
                self.fabric.print(
                    f"Epoch {epoch_num} Iteration {batch_idx} Loss: {loss.detach().item():.4f} "
                    f"LR: {metrics['lr']:.6f} Time: {metrics['iter_time']:.2f}s"
                )

            # Save the model checkpoint
            if batch_idx != 0 and (batch_idx % self.checkpoint_interval == 0 or batch_idx == len(train_dataloader)-1):
                checkpoint_path = pathlib.Path(self.checkpoint_dir) / f"lastest.ckpt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "iter_num": batch_idx + 1 if batch_idx < len(train_dataloader)-1 else 0,
                    "epoch_num": epoch_num if batch_idx < len(train_dataloader)-1 else epoch_num + 1,
                }
                if self.checkpoint_filter is not None:
                    self.fabric.save(checkpoint_path, state, filter={'model': self.checkpoint_filter})
                else:
                    self.fabric.save(checkpoint_path, state)
                self.fabric.print(f"Checkpoint saved at {checkpoint_path}")
                self.fabric.barrier()

                # Restore model from checkpoint
                torch.cuda.empty_cache()
                self.fabric.load(checkpoint_path, state, strict=False)
                model = state.pop("model")
                optimizer = state.pop("optimizer")
                scheduler = state.pop("scheduler")

                hprams = model.hprams
                if self.fabric.is_global_zero:
                    eval_model = AVAE(**hprams)
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    imcomplete_keys = eval_model.load_state_dict(state_dict['model'], strict=False)
                    print(f"Loaded model with missing keys: {imcomplete_keys}")
                    
                    # Style encoder evaluation
                    wrapped_encoder = WrappedEncoder(eval_model.style_encoder, num_gpus=self.fabric.world_size)
                    style_metrics = eval_hrs(model=wrapped_encoder,eval_batch_size=self.eval_batch_size)
                    self.fabric.print(f"Style encoder evaluation: {style_metrics}")
                    self.fabric.log_dict(style_metrics, step=total_steps)
                    with open(pathlib.Path(self.checkpoint_dir) / "style_encoder_eval.txt", "a") as f:
                        f.write(f"Results for step {total_steps}:\n")
                        for k, v in style_metrics.items():
                            f.write(f"{k}: {v}\n")
                    # TODO: Content encoder evaluation
                    # TODO: Generator evaluation
                    del wrapped_encoder
                    del eval_model
                    torch.cuda.empty_cache()
                    if style_metrics['avg/R@8'] > self.best_results.get('style_metrics', -1):
                        self.best_results['style_metrics'] = style_metrics['avg/R@8']
                        best_style_checkpoint_path = pathlib.Path(self.checkpoint_dir) / f"best_style_encoder.ckpt"
                        shutil.copy(checkpoint_path, best_style_checkpoint_path)
                self.fabric.barrier()

        return checkpoint_path
