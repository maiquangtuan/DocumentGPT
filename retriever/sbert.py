import os
import pickle
import time

import torch
from sentence_transformers import SentenceTransformer, util

class SemanticSearchSentenceTransformer:
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)


    def fit(self, data: list[str]):
        self.data = data
        self.data_embeddings = self.model.encode(data)

    def __call__(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.data_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            results.append(self.data[idx])

        return results

    





