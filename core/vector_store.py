import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions
from core.splitter import Chunk

# Persistent storage — disk pe save hoga
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

_client = None
_ef = None


def get_client():
    global _client
    if _client is None:
        try:
            # Local pe persistent, cloud pe in-memory
            _client = chromadb.PersistentClient(path=CHROMA_PATH)
        except Exception:
            # Streamlit Cloud — readonly filesystem
            _client = chromadb.EphemeralClient()
    return _client

def get_embedding_function():
    global _ef
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _ef


class VectorStore:
    """
    ChromaDB persistent vector store
    - Documents disk pe save hote hain
    - Session band ho toh bhi data rehta hai
    - 50+ pages easily handle karta hai
    """

    def __init__(self, collection_name: str = "documents"):
        self.client = get_client()
        self.ef = get_embedding_function()
        self.collection_name = collection_name

        # Collection get karo ya banao
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] Collection '{collection_name}' ready — {self.collection.count()} chunks stored")

    def add_chunks(self, chunks: list[Chunk], doc_id: str):
        """
        Chunks ko ChromaDB mein store karo
        doc_id — unique document identifier
        """
        if not chunks:
            return

        # Existing doc ke chunks delete karo (re-upload case)
        existing = self.collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self.collection.delete(where={"doc_id": doc_id})
            print(f"[VectorStore] Cleared {len(existing['ids'])} old chunks for doc: {doc_id}")

        ids = [f"{doc_id}_chunk_{c.index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "parent": c.parent,
                "index": c.index
            }
            for c in chunks
        ]

        # Batch mein add karo — large documents ke liye
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )

        print(f"[VectorStore] Added {len(chunks)} chunks for doc: {doc_id}")

    def search(self, query: str, top_k: int = 4, doc_id: str = None) -> list[dict]:
        """
        Semantic search — relevant chunks retrieve karo
        doc_id filter karo agar specific document se chahiye
        """
        where = {"doc_id": doc_id} if doc_id else None

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where
        )

        chunks = []
        if results["ids"][0]:
            for i, doc in enumerate(results["documents"][0]):
                chunks.append({
                    "text": doc,
                    "parent": results["metadatas"][0][i]["parent"],
                    "score": 1 - results["distances"][0][i],
                    "index": results["metadatas"][0][i]["index"]
                })

        return chunks

    def list_documents(self) -> list[str]:
        """Stored documents ki list"""
        all_meta = self.collection.get()["metadatas"]
        if not all_meta:
            return []
        doc_ids = list(set([m["doc_id"] for m in all_meta]))
        return doc_ids

    def delete_document(self, doc_id: str):
        """Document aur uske chunks delete karo"""
        self.collection.delete(where={"doc_id": doc_id})
        print(f"[VectorStore] Deleted doc: {doc_id}")

    def count(self) -> int:
        return self.collection.count()


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

    # Chunks banao
    chunks = chunk_document(source)
    print(f"Total chunks: {len(chunks)}")

    # Store karo
    store = VectorStore(collection_name="test")
    store.add_chunks(chunks, doc_id="eiffel_doc_1")

    # Search karo
    print("\nSearching: 'When was it built?'")
    results = store.search("When was it built?", top_k=2, doc_id="eiffel_doc_1")
    for r in results:
        print(f"  Score: {r['score']:.3f} — {r['text']}")

    print("\nSearching: 'How tall is the tower?'")
    results = store.search("How tall is the tower?", top_k=2, doc_id="eiffel_doc_1")
    for r in results:
        print(f"  Score: {r['score']:.3f} — {r['text']}")

    # Documents list
    print(f"\nStored documents: {store.list_documents()}")