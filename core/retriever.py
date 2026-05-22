import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.splitter import Chunk


_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading embedding model...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedder ready!")
    return _embedder


class HybridRetriever:
    """
    FAISS (semantic) + BM25 (keyword) hybrid search
    Dates, codes, names — kuch bhi miss nahi hoga
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]

        print("Building FAISS index...")
        embedder = get_embedder()
        embeddings = embedder.encode(self.texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        print("Building BM25 index...")
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        print("Retriever ready!\n")

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        embedder = get_embedder()

        # --- FAISS semantic search ---
        q_emb = embedder.encode([query])
        q_emb = np.array(q_emb).astype('float32')
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k * 2)
        
        faiss_scores = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                faiss_scores[idx] = float(score)

        # --- BM25 keyword search ---
        tokenized_query = query.lower().split()
        bm25_raw = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores 0-1
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1
        bm25_scores = {i: float(s / max_bm25) for i, s in enumerate(bm25_raw)}

        # --- Hybrid fusion (60% semantic + 40% keyword) ---
        all_ids = set(faiss_scores) | set(bm25_scores)
        hybrid = {}
        for i in all_ids:
            sem = faiss_scores.get(i, 0.0)
            kw = bm25_scores.get(i, 0.0)
            hybrid[i] = 0.6 * sem + 0.4 * kw

        # Top K chunks return karo
        top_ids = sorted(hybrid, key=lambda x: hybrid[x], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_ids]


if __name__ == "__main__":
    from core.splitter import chunk_document

    source = """
    The Eiffel Tower is located in Paris, France. It was built in 1889 
    by Gustave Eiffel. The tower stands 330 meters tall. It was originally 
    built as the entrance arch for the 1889 World Fair. More than 7 million 
    people visit it every year. The tower has three floors for visitors. 
    On the top floor there is a restaurant and a small apartment where 
    Gustave Eiffel used to receive guests. The tower was almost demolished 
    in 1909 but was saved because it was useful as a radio transmission tower.
    """

    chunks = chunk_document(source)
    retriever = HybridRetriever(chunks)

    # Test queries
    queries = [
        "When was the Eiffel Tower built?",
        "How tall is the tower?",
        "Who designed it?"
    ]

    for q in queries:
        print(f"Query: {q}")
        results = retriever.search(q, top_k=2)
        for r in results:
            print(f"  → {r.text}")
        print()