import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
from core.splitter import Chunk
from core.vector_store import VectorStore

class HybridRetriever:
    """
    ChromaDB (semantic) + BM25 (keyword) hybrid search
    - ChromaDB persistent hai — 50+ pages handle karta hai
    - BM25 dates, codes, names kabhi miss nahi karta
    """

    def __init__(self, chunks: list[Chunk], doc_id: str, store: VectorStore = None):
        self.chunks = chunks
        self.doc_id = doc_id
        self.texts = [c.text for c in chunks]

        # ChromaDB store
        self.store = store if store else VectorStore()
        self.store.add_chunks(chunks, doc_id=doc_id)

        # BM25 index
        print("[Retriever] Building BM25 index...")
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        print("[Retriever] Ready!\n")

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        # --- ChromaDB semantic search ---
        semantic_results = self.store.search(
            query=query,
            top_k=top_k * 2,
            doc_id=self.doc_id
        )
        semantic_scores = {}
        for r in semantic_results:
            semantic_scores[r["text"]] = {
                "score": r["score"],
                "parent": r["parent"]
            }

        # --- BM25 keyword search ---
        tokenized_query = query.lower().split()
        bm25_raw = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1
        bm25_scores = {
            self.texts[i]: float(s / max_bm25)
            for i, s in enumerate(bm25_raw)
        }

        # --- Hybrid fusion (60% semantic + 40% keyword) ---
        all_texts = set(semantic_scores.keys()) | set(bm25_scores.keys())
        hybrid = {}
        for text in all_texts:
            sem = semantic_scores.get(text, {}).get("score", 0.0)
            kw = bm25_scores.get(text, 0.0)
            hybrid[text] = {
                "hybrid_score": round(0.6 * sem + 0.4 * kw, 4),
                "parent": semantic_scores.get(text, {}).get("parent", text)
            }

        # Top K return karo
        top_texts = sorted(
            hybrid,
            key=lambda x: hybrid[x]["hybrid_score"],
            reverse=True
        )[:top_k]

        return [
            {
                "text": t,
                "parent": hybrid[t]["parent"],
                "score": hybrid[t]["hybrid_score"]
            }
            for t in top_texts
        ]


if __name__ == "__main__":
    from core.splitter import chunk_document

    source = """
    The Eiffel Tower is located in Paris, France. It was built in 1889 
    by Gustave Eiffel. The tower stands 330 meters tall. It was originally 
    built as the entrance arch for the 1889 World Fair. More than 7 million 
    people visit it every year. The tower has three floors for visitors.
    On the top floor there is a restaurant and a small apartment where 
    Gustave Eiffel used to receive guests.
    """

    chunks = chunk_document(source)
    retriever = HybridRetriever(chunks, doc_id="test_eiffel")

    queries = [
        "When was the Eiffel Tower built?",
        "How tall is the tower?",
        "Who designed it?"
    ]

    for q in queries:
        print(f"Query: {q}")
        results = retriever.search(q, top_k=2)
        for r in results:
            print(f"  Score: {r['score']} — {r['text'][:70]}...")
        print()