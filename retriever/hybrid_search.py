from .sbert import SemanticSearchSentenceTransformer 
from .bm25 import SemanticSearchBM25 

class SemanticSearchHybrid:
    
    def __init__(self, data: list[str], top_k: int = 6, sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.sentence_transformer = SemanticSearchSentenceTransformer(sbert_model_name)
        self.sentence_transformer.fit(data)
        self.bm25 = SemanticSearchBM25(data)
        self.top_k = top_k


    def __call__(self, query: str):
        sentence_transformer_results = self.sentence_transformer(query, self.top_k)
        bm25_results = self.bm25(query, self.top_k)
        results = list(set(sentence_transformer_results).intersection(bm25_results))


        return results
