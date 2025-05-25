import random
from typing import Any, List, Union
import re
import pandas as pd
import polars as pl
import datasets


def get_random_instance(data: List[Any], num_instance: int, generator: random.Random=None):
    if generator is None:
        generator = random.Random()
    if num_instance >= len(data):
        return data
    else:
        selected_index = generator.sample(list(range(len(data))), num_instance)
        return [data[idx] for idx in selected_index]


def get_embedding(model, text: Union[str, List[str]], batch_size: int = 32, max_length: int = 512):
    # model should have the method encode, that takes a list of strings and returns a list of embeddings
    assert hasattr(model, 'encode'), 'model should have the method encode to get embeddings'
    if isinstance(text, str):
        text = [text]
    embedding = model.encode(text, batch_size=batch_size, max_length=max_length)
    embedding = embedding.float()
    if len(embedding) == 1:
        return embedding[0]
    return embedding


def group_by_column(dataset: datasets.Dataset, column_to_group: str, columns_to_agg: List[str]):
    """
    Group the dataset by column_to_group and aggregate the columns in columns_to_agg as list
    Args:
        dataset: dataset to be grouped
        column_to_group: column to group the dataset
        columns_to_agg: columns to aggregate
    Returns:
        dataset: grouped dataset
    """
    print(f'Grouping dataset by {column_to_group} and aggregating {columns_to_agg} as list')
    # Remove all other columns
    columns_to_remove = set(dataset.column_names) - set([column_to_group] + columns_to_agg)
    dataset = dataset.remove_columns(columns_to_remove)
    df = dataset.to_pandas()
    df = pl.from_pandas(df)
    df = df.group_by(column_to_group).agg([pl.col(col) for col in columns_to_agg])
    # convert back to hf dataset
    print(f"Grouped dataset: {df}")
    # Convert back to pandas dataframe
    df = df.to_pandas()
    dataset = datasets.Dataset.from_pandas(df)
    return dataset
    # df = df.groupby(column_to_group).agg({
    #     col: list for col in columns_to_agg
    # }).reset_index()
    # return datasets.Dataset.from_pandas(df)


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
    # replace multiple tags with a single tag
    text = re.sub(r'(\[Tag\] )+', '[Tag] ', text)
    text = re.sub(r'(\[Email\] )+', '[Email] ', text)
    text = re.sub(r'(\[URL\] )+', '[URL] ', text)
    # Truncating too long text
    if len(text.split(' ')) > 2048:
        text = ' '.join(text.split(' ')[:2048])
    # remove special characters
    # text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()\-\[\]]', '', text)
    return text
    
    
def flatten(example, author_key, text_key):
    author_ids = []
    full_texts = []
    for author_id, author_data in zip(example[author_key], example[text_key]):
        for txt in author_data:
            author_ids.append(author_id)
            full_texts.append(txt)
    return {author_key: author_ids, text_key: full_texts}

