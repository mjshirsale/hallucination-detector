import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.splitter import chunk_document
from core.retriever import HybridRetriever
from core.vector_store import VectorStore
from core.nli_checker import check_claim
from core.llm_judge import judge_claim
from core.aggregator import aggregate, needs_llm


def get_doc_id(text: str) -> str:
    """Document ka unique ID generate karo — content hash se"""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def run_pipeline(
    source_text: str,
    llm_response: str,
    top_k: int = 3,
    progress_callback=None
) -> dict:
    """
    Full hallucination detection pipeline
    progress_callback — Streamlit progress bar ke liye
    """

    def update(msg: str, pct: float):
        if progress_callback:
            progress_callback(msg, pct)
        else:
            print(f"[{int(pct*100)}%] {msg}")

    update("Chunking source document...", 0.05)
    source_chunks = chunk_document(source_text)

    update("Building persistent vector store...", 0.15)
    doc_id = get_doc_id(source_text)
    store = VectorStore()

    update("Building hybrid retriever...", 0.25)
    retriever = HybridRetriever(source_chunks, doc_id=doc_id, store=store)

    update("Extracting claims from LLM response...", 0.35)
    response_chunks = chunk_document(llm_response)
    claims = list(set([c.text for c in response_chunks]))

    results = []
    llm_calls = 0
    total_claims = len(claims)

    for i, claim in enumerate(claims):
        progress = 0.35 + (0.55 * (i / total_claims))
        update(
            f"Analyzing claim {i+1}/{total_claims}: {claim[:50]}...",
            progress
        )

        # Relevant chunks retrieve karo
        relevant = retriever.search(claim, top_k=top_k)
        relevant_texts = [r["parent"] for r in relevant]

        # NLI — GPU pe
        nli_result = check_claim(claim, relevant_texts)

        # Smart filter — zaroorat ho tabhi Groq call
        llm_result = None
        if needs_llm(nli_result):
            llm_result = judge_claim(
                claim,
                relevant_texts,
                nli_result.get("evidence_chunk", "")
            )
            llm_calls += 1

        final = aggregate(nli_result, llm_result)
        results.append(final)

    update("Generating summary...", 0.95)

    hallucinated = [r for r in results if r["verdict"] == "Hallucinated"]
    uncertain = [r for r in results if r["verdict"] == "Uncertain"]
    supported = [r for r in results if r["verdict"] == "Supported"]

    # Density map — document ko sections mein divide karo
    section_size = max(1, len(results) // 5)
    density_map = []
    for i in range(0, len(results), section_size):
        section = results[i:i + section_size]
        hal_count = len([r for r in section if r["verdict"] == "Hallucinated"])
        density = round(hal_count / len(section) * 100, 1)
        density_map.append({
            "section": f"S{len(density_map)+1}",
            "density": density,
            "claims": len(section)
        })

    summary = {
        "total_claims": len(results),
        "hallucinated": len(hallucinated),
        "uncertain": len(uncertain),
        "supported": len(supported),
        "hallucination_rate": round(
            len(hallucinated) / len(results) * 100, 1
        ) if results else 0,
        "llm_calls_made": llm_calls,
        "llm_calls_saved": len(results) - llm_calls,
        "density_map": density_map,
        "doc_id": doc_id,
        "results": results
    }

    update("Done!", 1.0)
    return summary


if __name__ == "__main__":
    source = """
    The Eiffel Tower is located in Paris, France. It was built in 1889 
    by Gustave Eiffel. The tower stands 330 meters tall. It was originally 
    built as the entrance arch for the 1889 World Fair. More than 7 million 
    people visit it every year. The tower has three floors for visitors. 
    On the top floor there is a restaurant and a small apartment where 
    Gustave Eiffel used to receive guests. The tower was almost demolished 
    in 1909 but was saved because it was useful as a radio transmission tower.
    """

    llm_response = """
    The Eiffel Tower is located in Berlin, Germany. It was built in 1850 
    by Napoleon Bonaparte. The tower is 500 meters tall. It was built for 
    the 1889 World Fair. More than 7 million people visit it every year.
    """

    run_pipeline(source, llm_response)