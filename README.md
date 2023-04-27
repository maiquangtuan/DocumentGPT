
# DocumentGPT
### Why I create this Repo : 
I found many new products while chatting with colleagues, most of which are related to interacting with PDF files. During my search, I came across an online demo for PDF chatting at https://github.com/bhaskatripathi/pdfGPT. However, I believe I could create a better model using simple open-source modules like BM25 and SentenceTransformer.

The idea is similar to that of pdfGPT. We would first extract the text from the uploaded file and split it into chunks, which would be embedded into vectors. When a query is submitted, we would encode it into a vector and find the most relevant chunks based on a similarity search.

My implementation is based on the following references:
https://aidetic.in/blog/2020/07/18/lightning-fast-semantic-search-engine-using-bm25-and-neural-re-ranking/
https://www.sbert.net/docs/pretrained_models.html#semantic-search
https://github.com/dorianbrown/rank_bm25

For a more production-ready and standardized implementation of chatting with documents, please refer to https://github.com/MindPal-Space/docubot.









