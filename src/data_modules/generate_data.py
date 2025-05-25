import math
import random
from typing import Any, List, Union
import datasets
from vllm.sampling_params import GuidedDecodingParams

from src.data_modules.disentanglement_dataset import num_proc
from src.data_modules.templates import (
    CONTENT_COMPARITOR_SYSTEM_PROMPT,
    CONTENT_COMPARITOR_USER_PROMPT,
    ContentComparisonReply,
    GENERATE_STYLE_COUNTERFACTUAL_PROMPT, 
    STYLE_COUNTERFACTUAL_IN_CONTEXT_EXAMPLES, 
    StyleTransferReply,
    STYLE_COMPRARITOR_USER_PROMPT,
    STYLE_COMPRARITOR_SYSTEM_PROMPT,
    StyleComparisonReply,
    )
from src.model.blackbox_llm import BlackBoxIOSystem
from src.data_modules.utils import group_by_column, simple_preprocess


def generate_style_counterfactual_func(batch, model, stop_tokens, num_return, max_tokens):
    # Generate the style counterfactual text (rewrited text)
    input_text = batch['text_1'] # List of strings
    conversations = []
    for txt in input_text:
        messages = [
            {"role": "system", "content": GENERATE_STYLE_COUNTERFACTUAL_PROMPT,},
            {"role": "user", "content": txt},
        ]
        conversations.append(messages)
    json_schema = StyleTransferReply.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    # print(f"Conversations: {conversations}")
    output = model.generate(
        model_input=conversations,
        guided_decoding_params=guided_decoding_params,
        stop_tokens=stop_tokens,
        num_return=num_return,
        max_tokens=max_tokens,
    )
    rewrited_instance = {
        "text_2": [],
        "style_comparison": [],
    }
    for i in range(len(output)):
        try:
            instance = StyleTransferReply.model_validate_json(output[i][0])
            rewrited_instance["text_2"].append(instance.rewrited_text)
            rewrited_instance["style_comparison"].append(instance.style_comparison)
        except Exception as e:
            print(f"Error: {e}")
            rewrited_instance["text_2"].append("")
            rewrited_instance["style_comparison"].append("")
    return rewrited_instance

        
def generate_style_comparition_func(batch, model, stop_tokens, num_return, max_tokens):
    text_1 = batch['text_1']
    text_2 = batch['text_2']
    label = batch['label']
    comparisons = []
    conversations = []
    for txt1, txt2, lbl in zip(text_1, text_2, label):
        messages = [
            {"role": "system", "content": STYLE_COMPRARITOR_SYSTEM_PROMPT,},
            {"role": "user", "content": STYLE_COMPRARITOR_USER_PROMPT.format(text1=txt1, text2=txt2, label=lbl,)},
        ]
        conversations.append(messages)
    json_schema = StyleComparisonReply.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    output = model.generate(
        model_input=conversations,
        guided_decoding_params=guided_decoding_params,
        stop_tokens=stop_tokens,
        num_return=num_return,
        max_tokens=max_tokens,
    )
    for i in range(len(output)):
        try:
            instance = StyleComparisonReply.model_validate_json(output[i][0])
            comparisons.append(instance.style_comparison)
        except Exception as e:
            print(f"Error: {e}")
            comparisons.append("")
    return {'style_comparison': comparisons}


def generate_content_comparition_func(batch, model, stop_tokens, num_return, max_tokens):
    text_1 = batch['text_1']
    text_2 = batch['text_2']
    conversations = []
    for txt1, txt2 in zip(text_1, text_2):
        messages = [
            {"role": "system", "content": CONTENT_COMPARITOR_SYSTEM_PROMPT,},
            {"role": "user", "content": CONTENT_COMPARITOR_USER_PROMPT.format(text1=txt1, text2=txt2,)},
        ]
        conversations.append(messages)

    json_schema = ContentComparisonReply.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    output = model.generate(
        model_input=conversations,
        guided_decoding_params=guided_decoding_params,
        stop_tokens=stop_tokens,
        num_return=num_return,
        max_tokens=max_tokens,
    )
    comparisons = []
    lables = []
    for i in range(len(output)):
        try:
            instance = ContentComparisonReply.model_validate_json(output[i][0])
            comparisons.append(instance.content_comparison)
            lables.append(instance.determination)
        except Exception as e:
            print(f"Error: {e}")
            comparisons.append("")
            lables.append("")
    return {'content_comparison': comparisons, 'content_label': lables}


def is_query(example, num_positive):
    if example['positive'] is None:
        return False
    elif len(example['positive']) == 0:
        return False
    elif len(example['negative']) == 0:
        return False
    if len(example['positive']) >= num_positive:
        return True
    return False


def transform_format(example, data):
    positive_idx = example['positive'][0] # The first positive example, because we assume the positive examples are sorted by the similarity in ascending order
    negative_idx = example['negative'][0] # The first negative example, because we assume the negative examples are sorted by the similarity in descending order
    postive = data[positive_idx]
    negative = data[negative_idx]
    postive_txt = postive['fullText']
    negative_txt = negative['fullText']
    negative_author_id = negative['authorIDs']
    positive_cluster = postive['cluster']
    negative_cluster = negative['cluster']
    cluster = example['cluster']
    return {
        'authorIDs': example['authorIDs'],
        'fullText': example['fullText'],
        'positive': postive_txt,
        'negative': negative_txt,
        'negative_author_id': negative_author_id,
        'positive_cluster': positive_cluster,
        'negative_cluster': negative_cluster,
        'cluster': cluster,
    }


def convert_to_pairs(batch):
    full_text = batch['fullText']
    pos_txt = batch['positive']
    neg_txt = batch['negative']
    author_ids = batch['authorIDs']
    neg_author_ids = batch['negative_author_id']
    pos_cluster = batch['positive_cluster']
    neg_cluster = batch['negative_cluster']
    cluster = batch['cluster']
    pairs = {
        "text_1": [],
        "text_2": [],
        "label": [],
        "content_label": [],
        'cluster': [],
        "text_1_author_id": [],
        "text_2_author_id": [],
    }

    for i in range(len(full_text)):
        # Style-counterfactual instance (same content but different style)
        pairs["text_1"].append(pos_txt[i])
        pairs["text_1_author_id"].append(str(author_ids[i]))
        pairs["text_2"].append('')
        pairs["text_2_author_id"].append('Machine Generated')
        pairs["label"].append('different author')
        pairs['content_label'].append('same content')
        pairs['cluster'].append(pos_cluster[i])
        # Style-positive instance (same author)
        pairs["text_1"].append(full_text[i])
        pairs["text_1_author_id"].append(str(author_ids[i]))
        pairs["text_2"].append(pos_txt[i])
        pairs["text_2_author_id"].append(str(author_ids[i]))
        pairs["label"].append('same author')
        pairs["content_label"].append('')
        pairs['cluster'].append(cluster[i])
        # Style-negative instance (different author)
        pairs["text_1"].append(neg_txt[i])
        pairs["text_1_author_id"].append(str(neg_author_ids[i]))
        pairs["text_2"].append(full_text[i])
        pairs["text_2_author_id"].append(str(author_ids[i]))
        pairs["label"].append('different author')
        pairs["content_label"].append('')
        pairs['cluster'].append(neg_cluster[i])
    return pairs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate style counterfactual data")
    parser.add_argument("--data_name", type=str, required=True, help="The name of the dataset to use")

    args = parser.parse_args()

    data_name = args.data_name
    output_dir = 'data/preprocess/avae/' + data_name

    max_instances = 10000
    model_ckpt = "Qwen/QwQ-32B"
    seed = 42
    tensor_parallel_size = 1 # With Guided Decoding, the tensor_parallel_size should be 1 (this will be fixed in the future)
    half_precision = False
    temperature = 0.8
    top_p = 0.95
    top_k = 40
    max_num_seqs = 256
    stop_tokens = []
    num_return = 1
    max_tokens = 4096

    model = BlackBoxIOSystem(
        model_ckpt=model_ckpt,
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
        half_precision=half_precision,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_num_seqs=max_num_seqs,
    )

    data = datasets.load_dataset(data_name, split='train')
    data = data.map(lambda x, idx: {'is_query': is_query(x, 1), 'idx': idx}, num_proc=num_proc, with_indices=True)
    query_data = data.filter(lambda x: x['is_query'], num_proc=num_proc)
    query_idx = query_data['idx']
    cluster_info = query_data.remove_columns(list(set(query_data.column_names) - set(['idx', 'cluster'])))
    # Group the dataset by cluster
    cluster_info = group_by_column(cluster_info, 'cluster', ['idx'])
    cluster_info = {x['cluster']: x['idx'] for x in cluster_info}
    # Subsample the data to keep the cluster balanced
    number_per_cluster = math.ceil(len(query_idx) / len(cluster_info))
    cluster_info = {k: random.sample(v, min(len(v), number_per_cluster)) for k, v in cluster_info.items()}
    query_idx = sum(cluster_info.values(), [])
    query_idx = set(query_idx)
    # Get the data from the original dataset
    new_data = data.select(query_idx)
    new_data = new_data.map(lambda x: transform_format(x, data), num_proc=num_proc)
    remove_columns = set(new_data.column_names) - set(['text_1', 'text_2', 'label', 'content_label', 'text_1_author_id', 'text_2_author_id', 'cluster'])
    new_data = new_data.map(lambda x: convert_to_pairs(x), batched=True, remove_columns=remove_columns, num_proc=num_proc)
    selected_idx = range(len(new_data))
    cluster_info = new_data.map(lambda x, idx: {'cluster': x['cluster'], 'idx': idx}, num_proc=num_proc, with_indices=True, remove_columns=new_data.column_names)
    cluster_info = group_by_column(cluster_info, 'cluster', ['idx'])
    cluster_info = {x['cluster']: x['idx'] for x in cluster_info}
    if len(new_data) > max_instances:
        number_per_cluster = math.ceil(max_instances / len(cluster_info))
        cluster_info = {k: random.sample(v, min(len(v), number_per_cluster)) for k, v in cluster_info.items()}
        selected_idx = sum(cluster_info.values(), [])
    new_data = new_data.select(selected_idx)
    remove_columns = set(new_data.column_names) - set(['text_1', 'text_2', 'label', 'content_label', 'text_1_author_id', 'text_2_author_id'])
    new_data = new_data.remove_columns(remove_columns)
    new_data = new_data.map(lambda x: {'text_1': simple_preprocess(x['text_1']), 'text_2': simple_preprocess(x['text_2'])}, num_proc=num_proc)
    new_data = new_data.map(lambda x: {'style_comparison': '', 'content_comparison': ''}, num_proc=num_proc)

    # Generate the style counterfactual data (rewrited text)
    data_need_rewriting = new_data.filter(lambda x: x['text_2'] == '', num_proc=num_proc)
    data_need_rewriting = data_need_rewriting.map(lambda x: generate_style_counterfactual_func(x, model, stop_tokens, num_return, max_tokens), batched=True, batch_size=32)
    data_not_rewriting = new_data.filter(lambda x: x['text_2'] != '', num_proc=num_proc)
    data = datasets.concatenate_datasets([data_need_rewriting, data_not_rewriting])
    data = data.filter(lambda x: x['text_2'] != '')
    
    # Generate the style comparison data
    data_need_comparition = data.filter(lambda x: x['style_comparison'] == '', num_proc=num_proc)
    data_need_comparition = data_need_comparition.map(lambda x: generate_style_comparition_func(x, model, stop_tokens, num_return, max_tokens), batched=True, batch_size=32)
    data_not_comparition = data.filter(lambda x: x['style_comparison'] != '', num_proc=num_proc)
    data = datasets.concatenate_datasets([data_need_comparition, data_not_comparition])
    data = data.filter(lambda x: x['style_comparison'] != '')
    
    # Generate the content comparison data
    data_need_comparition = data.filter(lambda x: x['content_comparison'] == '', num_proc=num_proc)
    data_need_comparition = data_need_comparition.map(lambda x: generate_content_comparition_func(x, model, stop_tokens, num_return, max_tokens), batched=True, batch_size=32)
    data_not_comparition = data.filter(lambda x: x['content_comparison'] != '', num_proc=num_proc)
    data = datasets.concatenate_datasets([data_need_comparition, data_not_comparition])
    data = data.filter(lambda x: x['content_comparison'] != '')
    
    # Shuffle the data and save the data
    data = data.map(lambda x: {'text_1_author_id': f"{data_name}_{x['text_1_author_id']}", 'text_2_author_id': f"{data_name}_{x['text_2_author_id']}"}, num_proc=num_proc) 
    data = data.shuffle(seed=seed)

    data = data.filter(lambda example: example['label'] in ['different author', 'same author'], num_proc=20)
    data = data.filter(lambda example: example['content_label'] in ['different content', 'same content'], num_proc=20)

    data.save_to_disk(output_dir)
    



