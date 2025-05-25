import os
import re
import datasets
import torch
from transformers import HfArgumentParser

from src.models.author_reps_model import WrappedAuthorRepsModel
from src.args import DataArguments, ModelArguments, TrainingArguments


max_num_worker_suggest = 1
try:
    max_num_worker_suggest = len(os.sched_getaffinity(0))
except Exception:
    pass
num_proc = max_num_worker_suggest


def simple_preprocess(text: str) -> str:
    """
    Simple preprocess function to:
    - replace multiple spaces with a single space
    - replace multiple new lines with a new line
    - replace urls with the string "<url>"
    - replace emails with the string "<email>"
    - replace tags (e.g. @user) with the string "<tag>"
    - remove special characters
    Args:
        text: text to be preprocessed
    Returns:
        text: preprocessed text
    """
    # Replace multilple \r \t \f and spaces in to single character
    text = re.sub(r'[\r\t\f ]+', ' ', text)
    text = re.sub(r'&gt|&lt', ' ', text)
    # replace multiple new lines with a new line
    text = re.sub(r'\n+', '\n', text)
    # replace urls with the string "<url>"
    text = re.sub(r'http://\S+|https://\S+', '[URL]', text, flags=re.MULTILINE)
    # replace emails with the string "<email>"
    text = re.sub(r'\S+@\S+', '[Email]', text, flags=re.MULTILINE)
    # replace file paths with the string "<file>"
    # text = re.sub(r'([a-zA-Z]:\\|\\\\|\/)', '<file>', text, flags=re.MULTILINE)
    # replace tags (e.g. @user) with the string "<tag>"
    text = re.sub(r'@\w+', '[Tag]', text, flags=re.MULTILINE)
    # Truncating too long text
    if len(text.split(' ')) > 2048:
        text = ' '.join(text.split(' ')[:2048])
    # remove special characters
    # text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()\-\[\]]', '', text)
    return text

def retrieve_content_counterfactual_func(batch, model, threshold=0.5):
    assert batch['fullText'] == 1, 'The batch size should be 1'
    collection = batch[0]['fullText']
    # collection = [[txt] for txt in collection]
    style_embeddings = model.batch_encode(collection, batchsize=64).float().cpu() #[num_doc, embedding_dim]
    style_embeddings = torch.nn.functional.normalize(style_embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(style_embeddings, style_embeddings.T)
    min_sim = torch.min(sim_matrix)
    # Ignore the upper triangle of the similarity matrix to avoid duplicate pairs
    # and the diagonal to avoid self-similarity
    sim_matrix = torch.tril(sim_matrix, diagonal=0, out=1)
    threshold = min(threshold, min_sim)
    keep_mask = sim_matrix <= threshold # Only keep the documents with the style similarity less than the threshold to select the hard positive
    pairs = torch.nonzero(keep_mask, as_tuple=False)
    pairs = [(i, j) for i, j in pairs if i < j]
    instances = {
        'authorIDs': [],
        'fullText': [],
        'content_counterfactual': [],
    }
    for i, j in pairs:
        txt1 = collection[i]
        txt2 = collection[j]
        instances['authorIDs'].extend([batch[0]['authorIDs'], batch[0]['authorIDs']])
        instances['fullText'].extend([txt1, txt2])
        instances['content_counterfactual'].extend([txt2, txt1])
    return instances

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name', required=True, type=str, help='Name of the dataset'
    )
    args = parser.parse_args()

    config_file = "/home/hieum/uonlp/hiatus_main/checkpoints/gte-qwen2-v2/config.yaml"
    checkpoint_path = "/home/hieum/uonlp/hiatus_main/checkpoints/gte-qwen2-v2/best_model.ckpt"
    hf_parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    print(f"Loading yaml config {config_file}")
    data_args, model_args, training_args = hf_parser.parse_yaml_file(yaml_file=config_file)
    encoder_model = WrappedAuthorRepsModel(
        model_name_or_path=model_args.model_name_or_path,
        number_query_tokens=model_args.number_query_tokens,
        use_lora=model_args.use_lora,
        use_pissa=model_args.use_pissa,
        num_attention_heads=model_args.num_attention_heads,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.target_modules,
        adapter_name=model_args.adapter_name,
        quantization=training_args.quantization,
        attn_implementation=model_args.attn_implementation,
        pooling_method=model_args.pooling_method,
        backbone_type=model_args.backbone_type,
        model_checkpoint=checkpoint_path,
        is_bidirectional=model_args.is_bidirectional,
        num_gpus=16
    )
    encoder_model.eval()

    data = datasets.load_dataset(args.data_name, split='train')
    data = data.map(lambda x: {'fullText': [simple_preprocess(txt) for txt, _ in x['fullText']]}, num_proc=num_proc)
    collums_to_remove = set(data.column_names) - set(['authorIDs', 'fullText'])
    data = data.remove_columns(collums_to_remove)
    data = data.map(lambda x: retrieve_content_counterfactual_func(x, encoder_model), batch_size=1)
    breakpoint()



