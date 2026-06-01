import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
from core.splitter import chunk_document, _split_sentences
from core.retriever import HybridRetriever
from core.vector_store import VectorStore
from core.nli_checker import check_claim
from core.llm_judge import judge_claim
from core.aggregator import aggregate, needs_llm
from core.fact_checker import compare_facts


def get_doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def run_pipeline(
    source_text: str,
    llm_response: str,
    top_k: int = 3,
    progress_callback=None
) -> dict:

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

    # Smart claim extraction — no aggressive chunking
    raw_claims = _split_sentences(llm_response)

    # Filter — meaningful claims only (10-60 words)
    claims = [
        c for c in raw_claims
        if 8 < len(c.split()) < 60
    ]

    # Deduplicate
    claims = list(dict.fromkeys(claims))

    # Hard limit — max 20 claims
    claims = claims[:20]

    print(f"[Pipeline] {len(claims)} claims to analyze")

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
        source_context = " ".join(relevant_texts)

        # NLI check — GPU pe
        nli_result = check_claim(claim, relevant_texts)

        # Fact check — numbers, stats, orgs
        fact_result = compare_facts(source_context, claim)

        # Smart filter — dono signals consider karo
        llm_result = None
        if needs_llm(nli_result, fact_result):
            llm_result = judge_claim(
                claim,
                relevant_texts,
                nli_result.get("evidence_chunk", "")
            )
            llm_calls += 1

        # 3-signal aggregate
        final = aggregate(nli_result, llm_result, fact_result)
        results.append(final)

    update("Generating summary...", 0.95)

    hallucinated = [r for r in results if r["verdict"] == "Hallucinated"]
    uncertain = [r for r in results if r["verdict"] == "Uncertain"]
    supported = [r for r in results if r["verdict"] == "Supported"]

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