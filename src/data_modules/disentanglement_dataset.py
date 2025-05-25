from collections import defaultdict
import math
import os
import random
from typing import Any, Dict, List
import numpy as np
import torch
import datasets
from transformers import PreTrainedTokenizer, BatchEncoding


from src.data_modules.templates import (
    RECONSTRUCT_PROMPT,
    STYLE_REP_COMPARITOR_PROMPT,
    StyleRepComparisonReply,
    CONTENT_REP_COMPARITOR_PROMPT,
    ContentRepComparisonReply,
)


max_num_worker_suggest = 1
try:
    max_num_worker_suggest = len(os.sched_getaffinity(0))
except Exception:
    pass
num_proc = max_num_worker_suggest


class DisentanglementDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        num_train_example: int=None,
        placeholder: str='<|placeholder|> ',
        seed: int=777,
    ):
        super().__init__()
        self.data_name_or_path = data_name_or_path
        self.num_train_example = num_train_example
        self.seed = seed
        print('Processing data with seed {}'.format(self.seed))
        self.rnd = random.Random(seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.placeholder = placeholder
        self.tokenizer = tokenizer
        assert self.placeholder in self.tokenizer.get_vocab(), 'The placeholder should be in the tokenizer vocab'

        # Load the dataset
        dataset = datasets.load_dataset(data_name_or_path, split='train')
        if num_train_example is not None and num_train_example < len(dataset):
            dataset = dataset.train_test_split(test_size=len(dataset) - num_train_example, seed=self.seed)['train']
        self.dataset = dataset
    

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]

        txt1_reconstruction_message = [
            {'role': 'user', 'content': RECONSTRUCT_PROMPT.format(style_rep=self.placeholder, content_rep=self.placeholder)},
            {'role': 'assistant', 'content': example['text_1']},
        ]
        txt1_input = self.tokenizer.apply_chat_template(txt1_reconstruction_message, tokenize=False)
        txt1_prompt = self.tokenizer.apply_chat_template(txt1_reconstruction_message[:1], tokenize=True, return_tensors=None)
        txt1_prompt_length = len(txt1_prompt)

        txt2_reconstruction_message = [
            {'role': 'user', 'content': RECONSTRUCT_PROMPT.format(style_rep=self.placeholder, content_rep=self.placeholder)},
            {'role': 'assistant', 'content': example['text_2']},
        ]
        txt2_input = self.tokenizer.apply_chat_template(txt2_reconstruction_message, tokenize=False)
        txt2_prompt = self.tokenizer.apply_chat_template(txt2_reconstruction_message[:1], tokenize=True, return_tensors=None)
        txt2_prompt_length = len(txt2_prompt)
        
        assert example['label'] in ['different author', 'same author'], 'The label should be either "different author" or "same author"'
        assert example['content_label'] in ['different content', 'same content'], 'The content label should be either "different content" or "same content"'
        style_labels = 0 if example['label'] == 'different author' else 1
        content_labels = 0 if example['content_label'] == 'different content' else 1
        style_rep_compartion = StyleRepComparisonReply(
            determination=example['label'],
            explaination=example['style_comparison'],
        )
        content_rep_compartion = ContentRepComparisonReply(
            determination=example['content_label'],
            explaination=example['content_comparison'],
        )
        style_rep_compartion = style_rep_compartion.model_dump_json()
        content_rep_compartion = content_rep_compartion.model_dump_json()
        style_discriminator_message = [
            {'role': 'user', 'content': STYLE_REP_COMPARITOR_PROMPT.format(text1=self.placeholder, text2=self.placeholder)},
            {'role': 'assistant', 'content': style_rep_compartion},
        ]
        style_discriminator_input = self.tokenizer.apply_chat_template(style_discriminator_message, tokenize=False)
        style_discriminator_prompt = self.tokenizer.apply_chat_template(style_discriminator_message[:1], tokenize=True, return_tensors=None)
        style_discriminator_prompt_length = len(style_discriminator_prompt)
        content_discriminator_message = [
            {'role': 'user', 'content': CONTENT_REP_COMPARITOR_PROMPT.format(text1=self.placeholder, text2=self.placeholder)},
            {'role': 'assistant', 'content': content_rep_compartion},
        ]
        content_discriminator_input = self.tokenizer.apply_chat_template(content_discriminator_message, tokenize=False)
        content_discriminator_prompt = self.tokenizer.apply_chat_template(content_discriminator_message[:1], tokenize=True, return_tensors=None)
        content_discriminator_prompt_length = len(content_discriminator_prompt)
        return {
            'text_1': example['text_1'],
            'text_2': example['text_2'],
            'text1_reconstruct': txt1_input,
            'txt1_reconstruct_prompt_length': txt1_prompt_length,
            'text2_reconstruct': txt2_input,
            'txt2_reconstruct_prompt_length': txt2_prompt_length,
            'style_discriminator_input': style_discriminator_input,
            'style_discriminator_prompt_length': style_discriminator_prompt_length,
            'content_discriminator_input': content_discriminator_input,
            'content_discriminator_prompt_length': content_discriminator_prompt_length,
            'style_labels': style_labels,
            'content_labels': content_labels,
        }

class DisentanglementDatasetCollator:
    def __init__(
            self, 
            style_encoder_tokenizer: PreTrainedTokenizer,
            content_encoder_tokenizer: PreTrainedTokenizer,
            generator_tokenizer: PreTrainedTokenizer, 
            max_length: int=512, 
            placeholder_token_id: int=None, 
            placeholder: str='<|placeholder|> ',
            prompt_loss: bool=True,
            ):
        self.style_encoder_tokenizer = style_encoder_tokenizer
        self.content_encoder_tokenizer = content_encoder_tokenizer
        self.tokenizer = generator_tokenizer
        self.max_length = max_length
        self.placeholder_token_id = placeholder_token_id
        self.placeholder = placeholder
        assert self.placeholder_token_id is not None, 'The placeholder token id should be set'
        assert self.placeholder in self.tokenizer.get_vocab(), 'The placeholder should be in the tokenizer vocab'
        # The placeholder token id and placeholder token should be matched
        assert self.placeholder_token_id == self.tokenizer.convert_tokens_to_ids(self.placeholder), 'The placeholder token id and placeholder token should be matched'
        # If mask token is not in the tokenizer, add it as eso token
        if self.tokenizer.mask_token_id is None:
            self.tokenizer.mask_token = self.tokenizer.eos_token
            self.tokenizer.mask_token_id = self.tokenizer.eos_token_id
        self.prompt_loss = prompt_loss

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        txt_1 = [example['text_1'] for example in batch]
        txt_2 = [example['text_2'] for example in batch]
        txt1_reconstruct = [example['text1_reconstruct'] for example in batch]
        txt2_reconstruct = [example['text2_reconstruct'] for example in batch]
        style_discriminator_input = [example['style_discriminator_input'] for example in batch]
        content_discriminator_input = [example['content_discriminator_input'] for example in batch]
        # Get the prompt length
        txt1_reconstruct_prompt_length = [example['txt1_reconstruct_prompt_length'] for example in batch]
        txt2_reconstruct_prompt_length = [example['txt2_reconstruct_prompt_length'] for example in batch]
        style_discriminator_prompt_length = [example['style_discriminator_prompt_length'] for example in batch]
        content_discriminator_prompt_length = [example['content_discriminator_prompt_length'] for example in batch]

        # Tokenize the text
        style_encoder_input_tokenized = self.style_encoder_tokenizer(
            txt_1 + txt_2,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        content_encoder_input_tokenized = self.content_encoder_tokenizer(
            txt_1 + txt_2,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        # Tokenize the txt_1 and text_2 reconstruct
        txt = txt1_reconstruct + txt2_reconstruct
        txt_reconstruct_tokenized = self.tokenizer(
            txt,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        txt_placehoder_token_pos = (txt_reconstruct_tokenized['input_ids'] == self.placeholder_token_id).nonzero(as_tuple=True)
        # Replace the placeholder token id with mask token id
        txt_reconstruct_tokenized['input_ids'][txt_placehoder_token_pos] = self.tokenizer.mask_token_id
        reconstruct_labels = txt_reconstruct_tokenized['input_ids'].clone()
        padding_mask = txt_reconstruct_tokenized['attention_mask'] == 0
        # Set the padding tokens to -100 so they are ignored in the loss
        reconstruct_labels[padding_mask] = -100
        txt_reconstruct_tokenized['labels'] = reconstruct_labels
        if self.prompt_loss is False:
            # Set the prompt tokens to -100 so they are ignored in the loss
            for i, prompt_length in enumerate(txt1_reconstruct_prompt_length + txt2_reconstruct_prompt_length):
                if self.tokenizer.padding_side == 'right':
                    txt_reconstruct_tokenized['labels'][i, :prompt_length] = -100
                else:
                    # find the first non-padding token position
                    first_non_padding_pos = (txt_reconstruct_tokenized['input_ids'][i, :] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
                    # set the prompt tokens to -100 so they are ignored in the loss
                    txt_reconstruct_tokenized['labels'][i, first_non_padding_pos:prompt_length] = -100
                # Warning if all labels is -100
                if (txt_reconstruct_tokenized['labels'][i, :] == -100).all():
                    print(f"Warning: all input is -100 for example {i}.")
                    print(f"Input: {txt[i]}")

        # Tokenize the style_discriminator_input and find the placeholder tokens positions
        style_discriminator_input_tokenized = self.tokenizer(
            style_discriminator_input,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        style_placehoder_token_pos = (style_discriminator_input_tokenized['input_ids'] == self.placeholder_token_id).nonzero(as_tuple=True)
        # Replace the placeholder token id with mask token id
        style_discriminator_input_tokenized['input_ids'][style_placehoder_token_pos] = self.tokenizer.mask_token_id
        style_discriminator_labels = style_discriminator_input_tokenized['input_ids'].clone()
        padding_mask = style_discriminator_input_tokenized['attention_mask'] == 0
        style_discriminator_labels[padding_mask] = -100
        style_discriminator_input_tokenized['labels'] = style_discriminator_labels
        if self.prompt_loss is False:
            # Set the prompt tokens to -100 so they are ignored in the loss
            for i, prompt_length in enumerate(style_discriminator_prompt_length):
                if self.tokenizer.padding_side == 'right':
                    style_discriminator_input_tokenized['labels'][i, :prompt_length] = -100
                else:
                    # find the first non-padding token position
                    first_non_padding_pos = (style_discriminator_input_tokenized['input_ids'][i, :] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
                    # set the prompt tokens to -100 so they are ignored in the loss
                    style_discriminator_input_tokenized['labels'][i, first_non_padding_pos:prompt_length] = -100
                # Warning if all labels is -100
                if (style_discriminator_input_tokenized['labels'][i, :] == -100).all():
                    print(f"Warning: all input is -100 for example {i}.")
                    print(f"Input: {style_discriminator_input[i]}")

        # Tokenize the content_discriminator_input and find the placeholder tokens positions
        content_discriminator_input_tokenized = self.tokenizer(
            content_discriminator_input,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        content_placehoder_token_pos = (content_discriminator_input_tokenized['input_ids'] == self.placeholder_token_id).nonzero(as_tuple=True)
        # Replace the placeholder token id with mask token id
        content_discriminator_input_tokenized['input_ids'][content_placehoder_token_pos] = self.tokenizer.mask_token_id
        content_discriminator_labels = content_discriminator_input_tokenized['input_ids'].clone()
        padding_mask = content_discriminator_input_tokenized['attention_mask'] == 0
        content_discriminator_labels[padding_mask] = -100
        content_discriminator_input_tokenized['labels'] = content_discriminator_labels
        if self.prompt_loss is False:
            # Set the prompt tokens to -100 so they are ignored in the loss
            for i, prompt_length in enumerate(content_discriminator_prompt_length):
                if self.tokenizer.padding_side == 'right':
                    content_discriminator_input_tokenized['labels'][i, :prompt_length] = -100
                else:
                    # find the first non-padding token position
                    first_non_padding_pos = (content_discriminator_input_tokenized['input_ids'][i, :] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
                    # set the prompt tokens to -100 so they are ignored in the loss
                    content_discriminator_input_tokenized['labels'][i, first_non_padding_pos:prompt_length] = -100
                # Warning if all labels is -100
                if (content_discriminator_input_tokenized['labels'][i, :] == -100).all():
                    print(f"Warning: all input is -100 for example {i}.")
                    print(f"Input: {content_discriminator_input[i]}")
        
        return {
            'style_encoder_input_tokenized': style_encoder_input_tokenized,
            'content_encoder_input_tokenized': content_encoder_input_tokenized,
            'txt_reconstruct_tokenized': txt_reconstruct_tokenized,
            'txt_placeholder_token_pos': txt_placehoder_token_pos,
            'style_discriminator_input_tokenized': style_discriminator_input_tokenized,
            # Tuple of (tensor, tensor) with the positions of the placeholder tokens, with the first tensor being the batch index and the second tensor being the token index
            'style_placeholder_token_pos': style_placehoder_token_pos, 
            'content_discriminator_input_tokenized': content_discriminator_input_tokenized,
            # Tuple of (tensor, tensor) with the positions of the placeholder tokens, with the first tensor being the batch index and the second tensor being the token index
            'content_placeholder_token_pos': content_placehoder_token_pos,
            'style_labels': torch.tensor([example['style_labels'] for example in batch], dtype=torch.long),
            'content_labels': torch.tensor([example['content_labels'] for example in batch], dtype=torch.long),
        }

        

