from typing import List, Union
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.vae import VAE, VQVAE
from src.model.encoder import Encoder
from src.model.generator import Generator
from src.model.loss import AngleLoss


class AVAE(nn.Module):
    def __init__(
            self, 
            style_encoder_model_name_or_path: str,
            content_encoder_model_name_or_path: str,
            generator_model_name_or_path: str,
            embedding_dim: int = 1536,
            style_encoder_use_lora: bool = False,
            content_encoder_use_lora: bool = False,
            generator_use_lora: bool = False,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            target_modules: Union[str, List[str]] = "all",
            adapter_name: str = None,
            attn_implementation: str = None,
            pooling_method: str = 'mean',
            dropout_prob: float = 0.1,
            style_encoder_model_type: str = 'qwen2',
            content_encoder_model_type: str = 'qwen2',
            vae_loss_weight: float = 1.0,
            reconstruction_loss_weight: float = 1.0,
            style_discriminator_loss_weight: float = 1.0,
            content_discriminator_loss_weight: float = 1.0,
            token_mi_reg_weight: float = 0.0,
            mi_reg_weight: float = 0.0,
            use_vae: bool = True,
            style_loss_weight: float = 0.0,
            content_loss_weight: float = 0.0,
            constraint_loss_weight: float = 0.0,
            ):
        super().__init__()
        self.hprams = {
            'style_encoder_model_name_or_path': style_encoder_model_name_or_path,
            'content_encoder_model_name_or_path': content_encoder_model_name_or_path,
            'generator_model_name_or_path': generator_model_name_or_path,
            'style_encoder_use_lora': style_encoder_use_lora,
            'content_encoder_use_lora': content_encoder_use_lora,
            'generator_use_lora': generator_use_lora,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules,
            'adapter_name': adapter_name,
            'attn_implementation': attn_implementation,
            'pooling_method': pooling_method,
            'dropout_prob': dropout_prob,
            'style_encoder_model_type': style_encoder_model_type,
            'content_encoder_model_type': content_encoder_model_type,
            'vae_loss_weight': vae_loss_weight,
            'reconstruction_loss_weight': reconstruction_loss_weight,
            'style_discriminator_loss_weight': style_discriminator_loss_weight,
            'content_discriminator_loss_weight': content_discriminator_loss_weight,
            'token_mi_reg_weight': token_mi_reg_weight,
            'mi_reg_weight': mi_reg_weight,
            'use_vae': use_vae,
            'style_loss_weight': style_loss_weight,
            'content_loss_weight': content_loss_weight,
            'constraint_loss_weight': constraint_loss_weight,
        }
        self.embedding_dim = embedding_dim
        self.vae_loss_weight = vae_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.style_discriminator_loss_weight = style_discriminator_loss_weight
        self.content_discriminator_loss_weight = content_discriminator_loss_weight
        self.token_mi_reg_weight = token_mi_reg_weight
        self.mi_reg_weight = mi_reg_weight
        self.style_loss_weight = style_loss_weight
        self.content_loss_weight = content_loss_weight
        self.constraint_loss_weight = constraint_loss_weight

        self.style_encoder = Encoder(
            model_name_or_path=style_encoder_model_name_or_path,
            use_bidirectional=True,
            use_lora=style_encoder_use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            adapter_name=adapter_name,
            attn_implementation=attn_implementation,
            pooling_method=pooling_method,
            dropout_prob=dropout_prob,
            model_type=style_encoder_model_type,
            embedding_dim=self.embedding_dim,
            use_vae=use_vae,
        )
        self.content_encoder = Encoder(
            model_name_or_path=content_encoder_model_name_or_path,
            use_bidirectional=False, # will let AutoModel decide
            use_lora=content_encoder_use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            adapter_name=adapter_name,
            attn_implementation=attn_implementation,
            pooling_method=pooling_method,
            dropout_prob=dropout_prob,
            model_type=content_encoder_model_type,
            embedding_dim=self.embedding_dim,
            use_vae=use_vae,
        )

        self.content_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, 2),
        )
        self.style_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, 2),
        )
        self.ce = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='none')

        self.generator = Generator(
            model_name_or_path=generator_model_name_or_path,
            use_lora=generator_use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            adapter_name=adapter_name,
            attn_implementation=attn_implementation,
            input_embedding_dim=self.embedding_dim,
        )

    @property
    def style_encoder_hprams(self):
        return self.style_encoder.hprams
    
    @property
    def content_encoder_hprams(self):
        return self.content_encoder.hprams
    
    @property
    def generator_hprams(self):
        return self.generator.hprams
        
    def forward(
            self,
            style_encoder_inputs_ids: torch.Tensor, # [2*batch_size, seq_len]
            style_encoder_attention_mask: torch.Tensor, # [2*batch_size, seq_len]
            content_encoder_inputs_ids: torch.Tensor, # [2*batch_size, seq_len]
            content_encoder_attention_mask: torch.Tensor, # [2*batch_size, seq_len]
            reconstruct_txt_inputs_ids: torch.Tensor, # [2*batch_size, seq_len]
            reconstruct_txt_attention_mask: torch.Tensor, # [2*batch_size, seq_len]
            reconstruct_labels: torch.Tensor, # [2*batch_size, seq_len]
            txt_placeholder_token_pos,
            pref_style_hidden_states: torch.Tensor = None, # [2*batch_size, seq_len, embedding_dim]
            pref_content_hidden_states: torch.Tensor = None, # [2*batch_size, seq_len, embedding_dim]
            style_discriminator_input_ids: torch.Tensor = None, # [batch_size, seq_len]
            style_discriminator_attention_mask: torch.Tensor = None, # [batch_size, seq_len]
            style_discriminator_labels: torch.Tensor = None, # [batch_size, seq_len]
            style_placeholder_token_pos = None, 
            content_discriminator_input_ids: torch.Tensor = None, # [batch_size, seq_len]
            content_discriminator_attention_mask: torch.Tensor = None, # [batch_size, seq_len]
            content_discriminator_labels: torch.Tensor = None, # [batch_size, seq_len]
            content_placeholder_token_pos = None,
            style_labels: torch.Tensor = None, # [batch_size]
            content_labels: torch.Tensor = None, # [batch_size]
            kl_loss_weight: float = 1.0,
            ):
        assert style_encoder_inputs_ids.size(0) == content_encoder_inputs_ids.size(0), "Style and content encoder inputs must have the same batch size"
        bs = style_encoder_inputs_ids.size(0) // 2
        # Style Encoder
        style_encoder_outputs = self.style_encoder(
            input_ids=style_encoder_inputs_ids,
            attention_mask=style_encoder_attention_mask,
        )
        syle_vae_loss = style_encoder_outputs['vae_loss']
        style_reps = style_encoder_outputs['reps'] # [2*bs, embedding_dim]
        style_mu = style_encoder_outputs['mu'] # [2*bs, embedding_dim]
        style_mu = style_mu.clone().contiguous() # Prevent in-place operation and ensure contiguous memory
        txt1_style_reps, txt2_style_reps = style_mu.split(bs, dim=0) # [bs, embedding_dim]
        style_rep_loss = None
        if style_labels is not None and self.style_loss_weight > 0:
            _style_reps = torch.cat([txt1_style_reps, txt2_style_reps], dim=-1) # [bs, 2*embedding_dim]
            style_pred = self.style_classifier(_style_reps) # [bs, 2]
            style_rep_loss = self.ce(style_pred, style_labels.long())

        # Content Encoder
        content_encoder_outputs = self.content_encoder(
            input_ids=content_encoder_inputs_ids,
            attention_mask=content_encoder_attention_mask,
        )
        content_vae_loss = content_encoder_outputs['vae_loss']
        content_reps = content_encoder_outputs['reps']
        content_mu = content_encoder_outputs['mu'] # [2*bs, embedding_dim]
        content_mu = content_mu.clone().contiguous() # Prevent in-place operation and ensure contiguous memory
        txt1_content_reps, txt2_content_reps = content_mu.split(bs, dim=0)
        content_rep_loss = None
        if content_labels is not None and self.content_loss_weight > 0:
            _content_reps = torch.cat([txt1_content_reps, txt2_content_reps], dim=-1) # [bs, 2*embedding_dim]
            content_pred = self.content_classifier(_content_reps)
            content_rep_loss = self.ce(content_pred, content_labels.long())

        # VAE loss
        vae_loss = None
        if self.vae_loss_weight > 0:
            vae_loss = 0.0
            if syle_vae_loss is not None:
                vae_loss += syle_vae_loss
            if content_vae_loss is not None:
                vae_loss += content_vae_loss

        # Reconstruction loss
        reconstruct_output = self.generator(
            input_ids=reconstruct_txt_inputs_ids,
            attention_mask=reconstruct_txt_attention_mask,
            labels=reconstruct_labels,
            first_sentence_embedding=style_reps,
            second_sentence_embedding=content_reps,
            placeholder_token_pos=txt_placeholder_token_pos,
        )
        reconstruction_loss = reconstruct_output['loss']
        if self.token_mi_reg_weight > 0:
            # Make the distribution of the style and content token over generator vocab farther apart (KL divergence)
            style_token_logits = reconstruct_output['first_placeholder_token_logits'] # [bs, vocab_size]
            content_token_logits = reconstruct_output['second_placeholder_token_logits'] # [bs, vocab_size]
            # Compute KL distance between the two distributions
            style_token_log_probs = F.log_softmax(style_token_logits, dim=-1) # [bs, vocab_size]
            content_token_log_probs = F.log_softmax(content_token_logits, dim=-1) # [bs, vocab_size]
            kl_distance = torch.nn.KLDivLoss(reduction='mean', log_target=True)
            # Maximize the KL divergence between the style and content token distributions to encourage disentanglement
            token_mi_reg_loss = - kl_distance(style_token_log_probs, content_token_log_probs)
        else:
            token_mi_reg_loss = None
        
        # Style Discriminator
        style_discriminator_loss = None
        if self.style_discriminator_loss_weight > 0:
            style_discriminator_outputs = self.generator(
                input_ids=style_discriminator_input_ids,
                attention_mask=style_discriminator_attention_mask,
                labels=style_discriminator_labels,
                first_sentence_embedding=txt1_style_reps,
                second_sentence_embedding=txt2_style_reps,
                placeholder_token_pos=style_placeholder_token_pos,
            )
            style_discriminator_loss = style_discriminator_outputs['loss']

        # Content Discriminator
        content_discriminator_loss = None
        if self.content_discriminator_loss_weight > 0:
            content_discriminator_outputs = self.generator(
                input_ids=content_discriminator_input_ids,
                attention_mask=content_discriminator_attention_mask,
                labels=content_discriminator_labels,
                first_sentence_embedding=txt1_content_reps,
                second_sentence_embedding=txt2_content_reps,
                placeholder_token_pos=content_placeholder_token_pos,
            )
            content_discriminator_loss = content_discriminator_outputs['loss']

        constraint_loss = None
        if self.constraint_loss_weight > 0:
            # Style constraint loss (KL divergence between the style representation and the preferred style representation)
            if pref_style_hidden_states is not None:
                style_hidden = style_encoder_outputs['hidden_states'] # [2*bs, seq_len, embedding_dim]
                style_hidden = einops.rearrange(style_hidden, 'b s d -> (b s) d') # [2*bs*seq_len, embedding_dim]
                style_hidden = F.log_softmax(style_hidden, dim=-1) # [2*bs*seq_len, embedding_dim]
                pref_style_hidden = einops.rearrange(pref_style_hidden_states, 'b s d -> (b s) d')
                pref_style_hidden = F.softmax(pref_style_hidden, dim=-1)
                style_constraint_loss = self.kl_loss(style_hidden, pref_style_hidden) # [2*bs*seq_len, embedding_dim]
                # Ignore the padding tokens
                style_encoder_attention_mask = einops.rearrange(style_encoder_attention_mask, 'b s -> (b s)') # [2*bs*seq_len]
                style_encoder_attention_mask = style_encoder_attention_mask.unsqueeze(-1) # [2*bs*seq_len, 1]
                style_constraint_loss = style_constraint_loss * style_encoder_attention_mask
                style_constraint_loss = style_constraint_loss.sum() / style_encoder_attention_mask.sum() # Average over non-padding tokens
                constraint_loss = style_constraint_loss
            if pref_content_hidden_states is not None:
                content_hidden = content_encoder_outputs['hidden_states']
                content_hidden = einops.rearrange(content_hidden, 'b s d -> (b s) d') # [2*bs*seq_len, embedding_dim]
                content_hidden = F.log_softmax(content_hidden, dim=-1) # [2*bs*seq_len, embedding_dim]
                pref_content_hidden = einops.rearrange(pref_content_hidden_states, 'b s d -> (b s) d')
                pref_content_hidden = F.softmax(pref_content_hidden, dim=-1)
                content_constraint_loss = self.kl_loss(content_hidden, pref_content_hidden) # [2*bs*seq_len, embedding_dim]
                # Ignore the padding tokens
                content_encoder_attention_mask = einops.rearrange(content_encoder_attention_mask, 'b s -> (b s)') # [2*bs*seq_len]
                content_encoder_attention_mask = content_encoder_attention_mask.unsqueeze(-1) # [2*bs*seq_len, 1]
                content_constraint_loss = content_constraint_loss * content_encoder_attention_mask
                content_constraint_loss = content_constraint_loss.sum() / content_encoder_attention_mask.sum() # Average over non-padding tokens
                constraint_loss = constraint_loss + content_constraint_loss if constraint_loss is not None else content_constraint_loss

        # TODO: MI Mimimization with CLUB
        mi_loss = None

        loss = self.reconstruction_loss_weight * reconstruction_loss
        if self.vae_loss_weight > 0 and vae_loss is not None:
            loss = loss + self.vae_loss_weight * vae_loss * kl_loss_weight
        if style_discriminator_loss is not None:
            loss = loss + self.style_discriminator_loss_weight * style_discriminator_loss 
        if content_discriminator_loss is not None:
            loss = loss + self.content_discriminator_loss_weight * content_discriminator_loss 
        if token_mi_reg_loss is not None:
            loss = loss + self.token_mi_reg_weight * token_mi_reg_loss 
        if mi_loss is not None:
            loss = loss + self.mi_reg_weight * mi_loss 
        if style_rep_loss is not None:
            loss = loss + self.style_loss_weight * style_rep_loss 
        if content_rep_loss is not None:
            loss = loss + self.content_loss_weight * content_rep_loss 
        if constraint_loss is not None:
            loss = loss + self.constraint_loss_weight * constraint_loss
        
        return {
            'loss': loss,
            'vae_loss': vae_loss,
            'reconstruction_loss': reconstruction_loss,
            'style_discriminator_loss': style_discriminator_loss,
            'content_discriminator_loss': content_discriminator_loss,
            'token_mi_reg_loss': token_mi_reg_loss,
            'mi_loss': mi_loss,
            'style_rep_loss': style_rep_loss,
            'content_rep_loss': content_rep_loss,
            'constraint_loss': constraint_loss,
        }
    
    def save_style_encoder(self, path):
        self.style_encoder.save_pretrained(path)
    
    def save_content_encoder(self, path):
        self.content_encoder.save_pretrained(path)
    
    def save_generator(self, path):
        self.generator.save_pretrained(path)
        
        




            

            


            

