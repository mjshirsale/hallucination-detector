import re
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str          # chhota chunk — FAISS search ke liye
    parent: str        # pura parent paragraph — LLM ko bhejna ke liye
    index: int         # chunk number

def _split_sentences(text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]

def _make_paragraphs(sentences: list[str], size: int = 5) -> list[str]:
    """Sentences ko paragraphs mein group karo — parent chunks"""
    paragraphs = []
    for i in range(0, len(sentences), size):
        group = sentences[i:i + size]
        paragraphs.append(' '.join(group))
    return paragraphs

def _make_child_chunks(paragraph: str, window: int = 2, overlap: int = 1) -> list[str]:
    """Har paragraph se chhote overlapping chunks — FAISS search ke liye"""
    sentences = _split_sentences(paragraph)
    chunks = []
    step = max(1, window - overlap)
    for i in range(0, len(sentences), step):
        group = sentences[i:i + window]
        if group:
            chunks.append(' '.join(group))
    return chunks

def chunk_document(text: str) -> list[Chunk]:
    """
    Main function — poora document process karta hai
    Returns: list of Chunk objects (child chunk + uska parent)
    """
    sentences = _split_sentences(text)
    paragraphs = _make_paragraphs(sentences, size=5)
    
    all_chunks = []
    idx = 0
    for para in paragraphs:
        children = _make_child_chunks(para, window=2, overlap=1)
        for child in children:
            all_chunks.append(Chunk(
                text=child,
                parent=para,
                index=idx
            ))
            idx += 1
    return all_chunks


if __name__ == "__main__":
    sample = """
    The Eiffel Tower is located in Paris, France. It was built in 1889 
    by Gustave Eiffel. The tower stands 330 meters tall. It was originally 
    built as the entrance arch for the 1889 World Fair. More than 7 million 
    people visit it every year. The tower has three floors for visitors. 
    On the top floor there is a restaurant and a small apartment where 
    Gustave Eiffel used to receive guests.
    """
    
    chunks = chunk_document(sample)
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[Chunk {c.index}]")
        print(f"  Child : {c.text}")
        print(f"  Parent: {c.parent[:80]}...")
        print()