from typing import List
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []

    def build_index(self, documents: List[str]):
        self.corpus = documents
        embeddings = self.embedder.encode(documents, convert_to_tensor=False)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def query(self, text: str, top_k: int = 3) -> List[str]:
        if self.index is None:
            raise ValueError("Index not built")
        embedding = self.embedder.encode([text], convert_to_tensor=False)
        distances, indices = self.index.search(embedding, top_k)
        return [self.corpus[i] for i in indices[0]]
