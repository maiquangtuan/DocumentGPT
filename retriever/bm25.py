from rank_bm25 import BM25Okapi

class SemanticSearchBM25:
    
    def __init__(self, data: list[str]):
        self.data = data
        tokenized_corpus = [chunk.split(" ") for chunk in data]
        self.bm25_model = BM25Okapi(tokenized_corpus)
    
    def __call__(self, query: str, top_k):
        tokenized_query = query.split(" ")
        result = self.bm25_model.get_top_n(tokenized_query, self.data, n=top_k)
        return result
    
