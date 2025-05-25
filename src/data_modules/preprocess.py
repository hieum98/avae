import datasets
from datasets import load_dataset
import os
import shutil
import torch
import torch.nn as nn

from src.data_modules.style_dataset import PretrainStyleRepDataset, num_proc
from src.data_modules.utils import simple_preprocess


def preprocess_dataset(
    dataname: str,
    use_dense_retrieval_hard_negatives: bool = True,
    style_encoder = None,
    content_encoder = None,
    num_clusters: int = None,
    threshold: float = None,
    do_filter: bool = False,
    preprocess_data_dir: str = None,
    same_encoder: bool = False,
):
    """
    Preprocess the dataset by filtering, clustering, and finding positives and negatives.
    
    Args:
        dataname (str): The name of the dataset to preprocess.
        use_dense_retrieval_hard_negatives (bool): Whether to use dense retrieval for hard negatives.
        style_encoder: The encoder used for style representation.
        content_encoder: The encoder used for content representation.
        num_clusters (int): The number of clusters to use for clustering.
        threshold (float): The threshold for filtering easy documents.
        do_filter (bool): Whether to filter easy documents.

    Returns:
        dataset: The preprocessed dataset.
    """
    # Load the dataset
    dataset = load_dataset(dataname, split='train')
    dataset = dataset.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=num_proc)

    if 'documentID' not in dataset.column_names:
        dataset = dataset.map(lambda x, idx: {'documentID': idx}, with_indices=True, num_proc=num_proc)
    assert 'documentID' in dataset.column_names, 'The dataset should have documentID column! Please process to add the documentID column'
    assert 'authorIDs' in dataset.column_names, 'The dataset should have authorIDs column! Please process to add the authorIDs column'
    assert 'fullText' in dataset.column_names, 'The dataset should have fullText column! Please process to add the fullText column'

    if content_encoder is not None:
        if do_filter:
            if os.path.exists(os.path.join(preprocess_data_dir, f'{dataname}_filtered')):
                dataset = datasets.load_from_disk(os.path.join(preprocess_data_dir, f'{dataname}_filtered'))
                dataset = dataset.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=num_proc)
            else:
                # Filter the easy documents
                dataset = PretrainStyleRepDataset.filter_easy_documents(
                    dataset=dataset,
                    encoder=content_encoder,
                    threshold=0.8 
                )
                dataset.save_to_disk(os.path.join(preprocess_data_dir, f'{dataname}_filtered'))

        dataset = dataset.train_test_split(train_size=5000000, shuffle=True)['train']
        # Get the positives
        if os.path.exists(os.path.join(preprocess_data_dir, f'{dataname}_positives')):
            dataset = datasets.load_from_disk(os.path.join(preprocess_data_dir, f'{dataname}_positives'))
            dataset = dataset.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=num_proc)
        else:
            dataset = PretrainStyleRepDataset.get_positives(
                dataset=dataset,
                content_encoder=content_encoder,
                threshold=threshold,
            )
            if not same_encoder:
                collums_to_remove = set(dataset.column_names) - set(['authorIDs', 'documentID', 'fullText', 'positive', 'negative', 'cluster'])
            else:
                collums_to_remove = set(dataset.column_names) - set(['authorIDs', 'documentID', 'fullText', 'positive', 'negative', 'cluster', 'contentEmbedding'])
            dataset = dataset.remove_columns(collums_to_remove)
            dataset.save_to_disk(os.path.join(preprocess_data_dir, f'{dataname}_positives'))

    if num_clusters is not None and style_encoder is not None:
        # Cluster the dataset
        if os.path.exists(os.path.join(preprocess_data_dir, f'{dataname}_clustered')):
            dataset = datasets.load_from_disk(os.path.join(preprocess_data_dir, f'{dataname}_clustered'))
            dataset = dataset.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=num_proc)
        else:
            embed_col_name='contentEmbedding' if same_encoder else 'styleEmbedding'
            dataset = PretrainStyleRepDataset.cluster(dataset, style_encoder, number_cluster=num_clusters, embed_col_name=embed_col_name,)
            dataset.save_to_disk(os.path.join(preprocess_data_dir, f'{dataname}_clustered'))

    if use_dense_retrieval_hard_negatives and style_encoder is not None:
        # Find the hard negatives by using dense retrieval
        dataset = PretrainStyleRepDataset.get_negatives_by_dense_retriever(
            dataset=dataset,
            style_encoder=style_encoder,
            num_clusters=num_clusters,
            )
    elif not use_dense_retrieval_hard_negatives:
        # Find the hard negatives by using BM25
        dataset = PretrainStyleRepDataset.get_negatives_by_BM25(dataset=dataset, dataname=dataname,)
    
    assert 'positive' in dataset.column_names, 'The dataset should have positive column! Please process to get the positives by providing the content encoder'
    assert 'negative' in dataset.column_names, 'The dataset should have negative column! Please process to get the negatives by providing the style encoder or set use_dense_retrieval_hard_negatives to False'
    assert 'cluster' in dataset.column_names, 'The dataset should have cluster column! Please process to get the clusters by providing the style encoder and number of clusters'

    # Save the dataset to disk
    dataset.save_to_disk(os.path.join(preprocess_data_dir, dataname))
    dataset.cleanup_cache_files()
    # Remove all intermediate files/folders
    if os.path.exists(os.path.join(preprocess_data_dir, f'{dataname}_filtered')):
        shutil.rmtree(os.path.join(preprocess_data_dir, f'{dataname}_filtered'))
    if os.path.exists(os.path.join(preprocess_data_dir, f'{dataname}_positives')):
        shutil.rmtree(os.path.join(preprocess_data_dir, f'{dataname}_positives'))
    if os.path.exists(os.path.join(preprocess_data_dir, f'{dataname}_clustered')):
        shutil.rmtree(os.path.join(preprocess_data_dir, f'{dataname}_clustered'))


if __name__ == "__main__":
    import argparse
    import batched
    from angle_emb import AnglE

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, help='The name of the dataset to preprocess')
    parser.add_argument('--use_dense_retrieval_hard_negatives', action='store_true', help='Whether to use dense retrieval for hard negatives')
    parser.add_argument('--style_encoder_name_or_path', type=str, help='The style encoder to use', default=None)
    parser.add_argument('--num_clusters', type=int, help='The number of clusters to use', default=256)
    parser.add_argument('--threshold', type=float, help='The threshold for filtering easy documents', default=0.5)
    parser.add_argument('--do_filter', action='store_true', help='Whether to filter easy documents')
    parser.add_argument('--preprocess_data_dir', type=str, help='The directory to save the preprocessed dataset', default='data/preprocess')
    
    args = parser.parse_args()
    # check if cuda is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    content_encoder = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to(device)
    if torch.cuda.device_count() > 1:
        content_encoder.backbone = nn.DataParallel(content_encoder.backbone)
        content_encoder.pooler.model = nn.DataParallel(content_encoder.pooler.model)
    content_encoder.encode = batched.dynamically(content_encoder.encode, batch_size=256)
    if args.style_encoder_name_or_path is None:
        style_encoder = None
    else:
        raise NotImplementedError('Style encoder is not implemented yet')
    
    preprocess_dataset(
        dataname=args.dataname,
        use_dense_retrieval_hard_negatives=args.use_dense_retrieval_hard_negatives,
        style_encoder=style_encoder if style_encoder is not None else content_encoder,
        content_encoder=content_encoder,
        num_clusters=args.num_clusters,
        threshold=args.threshold,
        do_filter=args.do_filter,
        preprocess_data_dir=args.preprocess_data_dir,
        same_encoder=True
    )


