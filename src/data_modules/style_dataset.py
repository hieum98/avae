from collections import defaultdict
import math
import os
import random
from typing import Any, Dict, List
import numpy as np
import faiss
import torch
import datasets
import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding
# from retriv import SparseRetriever
import bm25s

from src.data_modules.utils import get_random_instance, get_embedding, group_by_column, simple_preprocess
from src.data_modules.templates import apply_template_for_rep_learning, tokenize_example

max_num_worker_suggest = 1
try:
    max_num_worker_suggest = len(os.sched_getaffinity(0))
except Exception:
    pass
num_proc = max_num_worker_suggest


class PretrainStyleRepDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_name_or_path: str,
        num_train_example: int=None,
        num_hard_negatives: int=32,
        num_positives: int=1,
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
        self.num_hard_negatives = num_hard_negatives
        self.num_positive = num_positives

        self.dataset, self.query_idx, self.cluster_info = self.load_data()

        assert self.num_hard_negatives < len(self.dataset), '# of hard negatives should be less than the total number of instances, but got {} and {} in {}'.format(self.num_hard_negatives, len(self.dataset), self.data_name_or_path)

    @staticmethod
    def filter_less_diverse_authors(dataset: datasets.Dataset, threshold: float=0.5):
        """
        Filter authors with two most disimilar embeddings are less than threshold. This is to remove authors containing only similar contents.
        Args:
            model: SentenceTransformer model that forcus on content embedding.
            dataset: dataset to be filtered with format {'authorIDs': str, 'documentID': str, 'fullText': str, 'contentEmbedding': np.array}
            threshold: threshold to filter authors, default 0.5. Lower threshold means more strict filter.
        Returns:
            dataset: filtered dataset with format {'authorIDs': str, 'documentID': str, 'fullText': str, 'contentEmbedding': np.array}
        """
        def is_easy_authors(example, threshold):
            content_embedding = example['contentEmbedding']
            content_embedding = torch.tensor(content_embedding) # (num_instance, embedding_dim)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(embeddings, embeddings.T)
            min_sim = torch.min(sim_matrix)
            return min_sim > threshold # if min_sim > threshold, then the author is easy
        collums_to_remove = set(dataset.column_names) - set(['authorIDs', 'contentEmbedding'])
        embedding_data = dataset.remove_columns(collums_to_remove)
        embedding_data = group_by_column(embedding_data, 'authorIDs', ['contentEmbedding'])
        embedding_data = embedding_data.map(lambda x: {'isEasy': is_easy_authors(x, threshold)})
        easy_authors = embedding_data.filter(lambda x: x['isEasy'])['authorIDs']
        dataset = dataset.filter(lambda x: x['authorIDs'] not in easy_authors, num_proc=num_proc, desc='Filtering easy authors')
        return dataset
    
    @staticmethod
    def filter_easy_documents(dataset: datasets.Dataset, encoder=None,  threshold: float=0.5):
        """
        Filter documents within the same author with the similarity of content embedding lagrer than threshold. This is to remove documents with similar content
        Args:
            model: SentenceTransformer model that forcus on content embedding.
            dataset: dataset to be filtered with format {'authorIDs': str, 'documentID': str, 'fullText': str, 'contentEmbedding': np.array}
            threshold: threshold to filter documents, default 0.5. Lower threshold means more strict filter.
        Returns:
            dataset: filtered dataset with format {'authorIDs': str, 'documentID': str, 'fullText': str, 'contentEmbedding': np.array}
        """
        def is_hard_documents(example, threshold):
            content_embedding = []
            doc_ids = []
            if len(example['contentEmbedding']) > 5000:
                print('There is one author with more than 5000 documents, we will only keep the first 5000 documents')
                example['contentEmbedding'] = example['contentEmbedding'][:5000]
            for emb in example['contentEmbedding']:
                embedding = emb['rep']
                doc_id = emb['doc_id']
                content_embedding.append(embedding)
                doc_ids.append(doc_id)
            if len(content_embedding) <= 1:
                return {'documentID': doc_ids}
            embeddings = torch.tensor(content_embedding)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(embeddings, embeddings.T)
            # Find the pair of documents with similarity smaller than threshold
            keep_mask = sim_matrix < threshold # (num_instance, num_instance)
            keep_mask = keep_mask.any(dim=0) # (num_instance)
            keep_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if keep_mask[i]]
            keep_doc_ids = list(set(keep_doc_ids))
            return {'documentID': keep_doc_ids}
        
        if 'contentEmbedding' not in dataset.column_names:
            print('Computing content embedding for filtering documents')
            dataset = dataset.map(lambda x: {'contentEmbedding': get_embedding(encoder, x['fullText'])}, batched=True, batch_size=32, num_proc=None,
                                  load_from_cache_file=True, cache_file_name='cache/content_embedding_{}'.format(dataset.info.dataset_name))

        collums_to_remove = set(dataset.column_names) - set(['authorIDs', 'contentEmbedding'])
        embedding_data = dataset.map(lambda x, idx: {'contentEmbedding': {'rep': x['contentEmbedding'], 'doc_id': idx}}, num_proc=num_proc, with_indices=True)
        embedding_data = embedding_data.remove_columns(collums_to_remove)
        embedding_data = group_by_column(embedding_data, 'authorIDs', ['contentEmbedding'])
        print('Filtering documents with similarity larger than {}'.format(threshold))
        embedding_data = embedding_data.map(lambda x: is_hard_documents(x, threshold), num_proc=None)
        keep_doc_ids = []
        for x in tqdm.tqdm(embedding_data, desc='Filtering documents'):
            for doc_id in x['documentID']:
                if doc_id not in keep_doc_ids:
                    keep_doc_ids.append(doc_id)
        del embedding_data # Free up memory
        
        dataset = dataset.select(keep_doc_ids)
        # dataset = dataset.filter(lambda x: x['documentID'] in keep_doc_ids, num_proc=num_proc)
        return dataset
    
    @staticmethod
    def cluster(dataset: datasets.Dataset, encoder=None, number_cluster: int=100, embed_col_name: str='styleEmbedding'):
        """
        Cluster the dataset based on the embeddings.
        Args:
            dataset: dataset to be clustered with format {'authorIDs': str, 'documentID': str, 'fullText': str, ...}
            encoder: SentenceTransformer model that focus on style embedding
            number_cluster: number of clusters to group the dataset
            embed_col_name: column name of the embeddings
        Returns:
            dataset: clustered dataset with format {'authorIDs': str, 'documentID': str, 'fullText': str, 'cluster': int, embed_col_name: np.array, ...}
        """
        if embed_col_name not in dataset.column_names:
            print('Computing style embedding for clustering')
            dataset = dataset.map(lambda x: {embed_col_name: get_embedding(encoder, x['fullText'])}, batched=True, batch_size=256, num_proc=None)
        data_size = len(dataset)
        if data_size < number_cluster * 20:
            ncentroids = data_size // 20
        else:
            ncentroids = number_cluster
        embeddings = np.array(dataset[embed_col_name])
        d = embeddings.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=100, verbose=True, min_points_per_centroid=20)
        kmeans.train(embeddings)
        _, I = kmeans.index.search(embeddings, 1)
        print('Assigning clusters')
        dataset = dataset.map(lambda x, idx: {"cluster": int(I[idx])}, with_indices=True, num_proc=num_proc)
        return dataset
    
    @staticmethod
    def get_positives(dataset: datasets.Dataset, content_encoder, threshold: float=0.5):
        """
        Get positive instances for document. Positive instances are the instances within the same author and in the order of decreasing content similarity.
        Args:
            dataset: dataset to get positive instances with format {'authorIDs': str, 'documentID': str, 'fullText': str}
            content_encoder: SentenceTransformer model that focus on content embedding
        Returns:
            dataset: dataset with positive instances with format {'authorIDs': str, 'documentID': str, 'fullText': str, 'positive': List[int]}
        """
        def sort_by_similarity(example, threshold):
            assert len(example['contentEmbedding']) == 1, 'The batchsize should be 1 to do flatten the dataset'
            if len(example['contentEmbedding'][0]) > 5000:
                print('The number of content embedding is larger than 5000, we will only keep the first 5000')
                example['contentEmbedding'][0] = example['contentEmbedding'][0][:5000]
            content_embedding = []
            doc_ids = []
            idxs = []
            for emb in example['contentEmbedding'][0]:
                embedding = emb['reps']
                doc_id = emb['doc_id']
                idx = emb['idx']
                content_embedding.append(embedding)
                doc_ids.append(doc_id)
                idxs.append(idx)
            embeddings = torch.tensor(content_embedding)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sim_matrix = torch.matmul(embeddings, embeddings.T)

            all_idx = []
            sorted_pos = []
            # for i in range(len(doc_ids)):
            for i in tqdm.tqdm(range(len(doc_ids)), desc='Sorting positive instances', disable=len(doc_ids) < 1000):
                sim = sim_matrix[i] # (num_instance)
                sorted_score, sorted_idx = torch.sort(sim, descending=False) # Sort from smallest to largest
                selected_idx = sorted_score <= threshold
                # Only keep the indices with similarity smaller than threshold
                sorted_idxs = [idxs[sorted_idx[j]] for j in range(len(doc_ids)) if selected_idx[j] and idxs[sorted_idx[j]] != idxs[i]]
                all_idx.append(idxs[i])
                if len(sorted_idxs) == 0:
                    sorted_idxs = [-1]
                sorted_pos.append(sorted_idxs)
            return {'idx': all_idx, 'positive': sorted_pos}
        
        if 'contentEmbedding' not in dataset.column_names:
            print('Computing content embedding')
            dataset = dataset.map(lambda x: {'contentEmbedding': get_embedding(content_encoder, x['fullText'])}, batched=True, batch_size=256, num_proc=None)
        
        collums_to_remove = set(dataset.column_names) - set(['authorIDs', 'contentEmbedding'])
        embedding_data = dataset.map(lambda x, idx: {'contentEmbedding': {'reps':x['contentEmbedding'], 'doc_id':x['documentID'], 'idx':idx}}, 
                                     with_indices=True, num_proc=num_proc)
        embedding_data = embedding_data.remove_columns(collums_to_remove)
        embedding_data = group_by_column(embedding_data, 'authorIDs', ['contentEmbedding'])
        print('Sorting positive instances')
        embedding_data = embedding_data.map(lambda x: sort_by_similarity(x, threshold), remove_columns=embedding_data.column_names, batched=True, batch_size=1)
        positives = {}
        for x in tqdm.tqdm(embedding_data, desc='Computing the positive dict'):
            if x['positive'] != [-1]:
                positives.update({x['idx']: x['positive']})
        print('Assigning positive instances')
        dataset = dataset.map(lambda x, idx: {'positive': positives.get(idx, None)}, with_indices=True)
        return dataset
    
    @staticmethod
    def get_negatives_by_BM25(dataset: datasets.Dataset, dataname: str):
        """
        Get negative instances for document. Negative instances are the instances with different author and most similar to the document by BM25.
        Args:
            dataset: dataset to be preprocessed with format {'authorIDs': str, 'documentID': str, 'fullText': str, ...}
            dataname: name of the dataset to store the BM25 index
        Returns:
            dataset: preprocessed dataset with format {'authorIDs': str, 'documentID': str, 'fullText': str, 'negative': List[int], ...}
        """
        def retrieve_negative(batch, retriever: bm25s.BM25):
            queries = []
            queries_ids = []
            for i in range(len(batch['fullText'])):
                if batch['positive'][i] is None:
                    continue
                if batch['positive'][i] == [-1]:
                    continue
                if len(batch['positive'][i]) == 0:
                    continue
                queries.append(batch['fullText'][i])
                queries_ids.append(i)
            if len(queries) == 0:
                return {'negative': [[-1]] * len(batch['fullText'])}
            queries_tokenized = bm25s.tokenize(queries, lower=False, stopwords=None, show_progress=False)
            results, scores = retriever.retrieve(queries_tokenized, k=1024, show_progress=False)
            retrieval_idx = []
            index = 0
            for i in range(len(batch['fullText'])):
                if i not in queries_ids:
                    retrieval_idx.append([-1])
                    continue
                _retrieval_idx = []
                highest_score = scores[index][0]
                for j in range(results.shape[1]):
                    if scores[index][j] / highest_score < 0.8:
                        _retrieval_idx.append(int(results[index][j]))
                retrieval_idx.append(_retrieval_idx)
                index += 1
            return {'negative': retrieval_idx}
        dataset = dataset.map(lambda x, idx: {'idx': idx}, with_indices=True, num_proc=num_proc)
        author_info = dataset.remove_columns(list(set(dataset.column_names) - set(['authorIDs', 'idx'])))
        author_info = group_by_column(author_info, 'authorIDs', ['idx'])
        author_info = {x['authorIDs']: x['idx'] for x in author_info}

        if os.path.exists('cache/bm25s_{}'.format(dataname)):
            print("Loading BM25 index from cache")
            retriever = bm25s.BM25.load('cache/bm25s_{}'.format(dataname), mmap=False)
            retriever.backend = 'numba'
        else:
            corpus = dataset['fullText']
            corpus_tokens = bm25s.tokenize(corpus, lower=False, stopwords=None)
            retriever = bm25s.BM25(backend='numba')
            retriever.index(corpus_tokens)
            retriever.save('cache/bm25s_{}'.format(dataname))

        print("Retrieving negatives...")
        dataset = dataset.map(lambda x: retrieve_negative(x, retriever=retriever), batched=True, batch_size=1024, num_proc=20)
        dataset = dataset.map(lambda x: {'negative': [idx for idx in x['negative'] if idx not in author_info[x['authorIDs']]] if x['negative'] != [-1] else None}, num_proc=num_proc)
        remove_columns = set(dataset.column_names) - set(['authorIDs', 'documentID', 'fullText', 'negative', 'positive', 'cluster'])
        dataset = dataset.remove_columns(remove_columns)
        return dataset
    
    @staticmethod
    def get_negatives_by_dense_retriever(dataset: datasets.Dataset, style_encoder, num_clusters: int=100):
        """
        Get negative instances for document. Negative instances are the instances with different author and in the same cluster of style embedding.
        Args:
            dataset: dataset to be preprocessed with format {'authorIDs': str, 'documentID': str, 'fullText': str, ...}
            style_encoder: SentenceTransformer model that focus on style embedding
            number_clusters: number of clusters to group style embeddings
        Returns:
            dataset: preprocessed dataset with format {'authorIDs': str, 'documentID': str, 'fullText': str, 'negative': List[int], ...}
        """
        if 'cluster' not in dataset.column_names:
            dataset = PretrainStyleRepDataset.cluster(dataset, style_encoder, number_cluster=num_clusters)
        dataset = dataset.map(lambda x, idx: {'idx': idx}, with_indices=True, num_proc=num_proc)
        author_info = dataset.remove_columns(list(set(dataset.column_names) - set(['authorIDs', 'idx'])))
        author_info = group_by_column(author_info, 'authorIDs', ['idx'])
        author_info = {x['authorIDs']: x['idx'] for x in author_info}
        cluster_info = dataset.remove_columns(list(set(dataset.column_names) - set(['cluster', 'idx'])))
        cluster_info = group_by_column(cluster_info, 'cluster', ['idx'])
        cluster_info = {x['cluster']: x['idx'] for x in cluster_info}
        # Randomly sample 2048 instances from the cluster
        dataset = dataset.map(lambda x: {'negative': random.sample(cluster_info[x['cluster']], k=2048)}, num_proc=num_proc)
        # Remove the positive instances from the negative instances
        dataset = dataset.map(lambda x: {'negative': list(set(x['negative']) - set(author_info[x['authorIDs']]))}, num_proc=num_proc)
        remove_columns = set(dataset.column_names) - set(['authorIDs', 'documentID', 'fullText', 'negative', 'positive', 'cluster'])
        dataset = dataset.remove_columns(remove_columns)
        return dataset

    def load_data(self):
        try:
            dataset = datasets.load_dataset(self.data_name_or_path, split='train')
        except:
            dataset = datasets.load_from_disk(self.data_name_or_path)

        dataset = dataset.map(
                lambda x: {'authorIDs': f"{self.data_name_or_path}-{x['authorIDs']}"}, 
                num_proc=num_proc,
                )
        dataset = dataset.map(lambda x: {'fullText': simple_preprocess(x['fullText'])}, num_proc=num_proc)
        def is_query(example, num_positive):
            if example['positive'] is None:
                return False
            if len(example['positive']) >= num_positive:
                return True
            return False

        dataset = dataset.map(lambda x, idx: {'is_query': is_query(x, self.num_positive), 'idx': idx}, num_proc=num_proc, with_indices=True)
        query_data = dataset.filter(lambda x: x['is_query'], num_proc=num_proc)
        query_idx = query_data['idx']
        cluster_info = query_data.remove_columns(list(set(query_data.column_names) - set(['idx', 'cluster'])))
        # Group the dataset by cluster
        cluster_info = group_by_column(cluster_info, 'cluster', ['idx'])
        cluster_info = {x['cluster']: x['idx'] for x in cluster_info}

        if self.num_train_example is None:
            self.num_train_example = len(query_idx)
        else:
            self.num_train_example = min(self.num_train_example, len(query_idx))

        # Subsample the data to keep the cluster balanced
        if len(query_idx) > self.num_train_example:
            number_per_cluster = math.ceil(self.num_train_example / len(cluster_info))
            cluster_info = {k: self.rnd.sample(v, min(len(v), number_per_cluster)) for k, v in cluster_info.items()}
            query_idx = sum(cluster_info.values(), [])
        
        # Sort the query_idx and cluster_info
        query_idx = sorted(query_idx)
        idx_dict = {idx: i for i, idx in enumerate(query_idx)}
        cluster_info = {k: [idx_dict[idx] for idx in v] for k, v in cluster_info.items()}
        cluster_info = {k: sorted(v) for k, v in cluster_info.items()}

        return dataset, query_idx, cluster_info
        
    def get_hard_negatives(self, example):
        author_id = example['authorIDs']
        if len(example['negative']) < self.num_hard_negatives:
            negative_idx = example['negative']
        else:
            negative_idx = list(self.rnd.sample(example['negative'], self.num_hard_negatives//2))
            negative_idx = negative_idx + example['negative'][:self.num_hard_negatives//2]
            negative_idx = list(set(negative_idx))
        
        negatives = []
        for idx in negative_idx:
            negatives.append(self.dataset[idx])
        # Remove the negative instances if they are same author as the query
        negatives = [x for x in negatives if x['authorIDs'] != author_id]
        # Add random instances with difference author if the number of negatives is less than self.num_hard_negatives
        while len(negatives) < self.num_hard_negatives:
            random_idx = self.rnd.choice(range(len(self.dataset)))
            if self.dataset[random_idx]['authorIDs'] != author_id:
                negatives.append(self.dataset[random_idx])
        assert len(negatives) == self.num_hard_negatives, 'The number of hard negatives should be equal to {}, but got {} in {}'.format(self.num_hard_negatives, len(negatives), self.data_name_or_path)
        return negatives
    
    def get_positive(self, example):
        positive_idx = example['positive']
        assert len(positive_idx) >= self.num_positive, 'The number of positive per query should be smaller than {}'.format(len(positive_idx))
        if self.rnd.random() < 0.5:
            positive_idx = self.rnd.sample(positive_idx, self.num_positive)
        else:
            positive_idx = positive_idx[:self.num_positive] # Always take instances by the order because we assume the positive instances are sorted by similarity
        positives = [self.dataset[idx] for idx in positive_idx]
        return positives

    def __len__(self):
        return len(self.query_idx)

    def __getitem__(self, idx):
        query_idx = self.query_idx[idx]
        query = self.dataset[query_idx]
        positives = self.get_positive(query)
        negatives = self.get_hard_negatives(query)

        author_id = [query['authorIDs']] + [x['authorIDs'] for x in positives + negatives]
        text = [query['fullText']] + [x['fullText'] for x in positives + negatives]

        return {
            'authorIDs': author_id,
            'text': text,
        }
        

class PretrainStyleRepCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, author_id_dict: Dict[str, int], max_seq_length: int=512, model_type: str=None):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.author_id_dict = author_id_dict
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> BatchEncoding:
        author_ids = []
        texts = []
        for item in batch:
            author_ids.extend(item['authorIDs'])
            texts.extend(item['text'])
        author_ids = [self.author_id_dict[x] for x in author_ids]
        author_ids = torch.tensor(author_ids, dtype=torch.long) # (bs*(1+num_positive+num_hard_negatives),)

        texts = apply_template_for_rep_learning(texts, self.model_type)
        model_inputs = tokenize_example(texts, self.tokenizer, self.max_seq_length)

        return {
            'input_ids': model_inputs['input_ids'], # (bs*(1+num_positive+num_hard_negatives), max_seq_length)
            'attention_mask': model_inputs['attention_mask'],
            'labels': author_ids, # (bs*(1+num_positive+num_hard_negatives),)
        }


        
        
        


       