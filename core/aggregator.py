def aggregate(nli_result: dict, llm_result: dict = None) -> dict:
    """
    Smart aggregator:
    - Sirf DeBERTa pe based verdict deta hai (fast)
    - Agar DeBERTa ne contradiction ya low confidence detect ki
      tabhi LLM result consider karta hai
    - Human readable explanation generate karta hai
    """

    nli_label = nli_result["nli_label"].upper()
    nli_score = nli_result["nli_score"]
    claim = nli_result["claim"]

    # --- Case 1: DeBERTa confident hai — LLM ki zaroorat nahi ---
    if nli_label == "ENTAILMENT" and nli_score >= 0.85:
        return {
            "claim": claim,
            "verdict": "Supported",
            "hallucination_score": 0.0,
            "confidence": "HIGH",
            "explanation": "Source text directly supports this claim.",
            "correction": "N/A",
            "reasoning": "",
            "evidence": nli_result.get("evidence_chunk", ""),
            "llm_used": False
        }

    if nli_label == "CONTRADICTION" and nli_score >= 0.85 and llm_result is None:
        return {
            "claim": claim,
            "verdict": "Hallucinated",
            "hallucination_score": 1.0,
            "confidence": "HIGH",
            "explanation": f"Source text directly contradicts this claim.",
            "correction": "N/A",
            "reasoning": "",
            "evidence": nli_result.get("evidence_chunk", ""),
            "llm_used": False
        }

    # --- Case 2: LLM result available hai — dono signals fuse karo ---
    if llm_result:
        nli_hal = 1.0 if nli_label == "CONTRADICTION" else 0.0
        llm_hal = 1.0 if llm_result["llm_verdict"] == "HALLUCINATED" else 0.0

        # NLI 60%, LLM 40%
        final_score = round(0.6 * nli_hal + 0.4 * llm_hal, 2)

        if final_score >= 0.6:
            verdict = "Hallucinated"
        elif final_score >= 0.3:
            verdict = "Uncertain"
        else:
            verdict = "Supported"

        # Conflict detect karo
        conflict = (nli_label == "CONTRADICTION") != (llm_result["llm_verdict"] == "HALLUCINATED")
        
        explanation = llm_result.get("reasoning", "")
        if conflict:
            explanation = f"[Models disagree] {explanation}"

        return {
            "claim": claim,
            "verdict": verdict,
            "hallucination_score": final_score,
            "confidence": llm_result.get("confidence", "MEDIUM"),
            "explanation": explanation,
            "correction": llm_result.get("correction", "N/A"),
            "reasoning": llm_result.get("reasoning", ""),
            "evidence": nli_result.get("evidence_chunk", ""),
            "llm_used": True
        }

    # --- Case 3: Neutral ya low confidence — uncertain ---
    return {
        "claim": claim,
        "verdict": "Uncertain",
        "hallucination_score": 0.5,
        "confidence": "LOW",
        "explanation": "Neither strong support nor contradiction found in source.",
        "correction": "N/A",
        "reasoning": "",
        "evidence": nli_result.get("evidence_chunk", ""),
        "llm_used": False
    }


def needs_llm(nli_result: dict) -> bool:
    """
    Smart filter — decide karta hai ki LLM call karni chahiye ya nahi
    Groq rate limits se bachata hai
    """
    label = nli_result["nli_label"].upper()
    score = nli_result["nli_score"]

    # DeBERTa confident hai — LLM ki zaroorat nahi
    if label == "ENTAILMENT" and score >= 0.85:
        return False

    # DeBERTa contradiction pe bohot confident hai — LLM ki zaroorat nahi
    if label == "CONTRADICTION" and score >= 0.90:
        return False

    # Baaki sab cases mein LLM se verify karo
    return True


if __name__ == "__main__":
    # Test Case 1: DeBERTa confident — LLM nahi lagega
    nli1 = {
        "claim": "Eiffel Tower is in Paris.",
        "nli_label": "entailment",
        "nli_score": 0.998,
        "is_hallucinated": False,
        "evidence_chunk": "The Eiffel Tower is located in Paris, France."
    }

    # Test Case 2: Contradiction — LLM lagega
    nli2 = {
        "claim": "Eiffel Tower is in Berlin.",
        "nli_label": "contradiction",
        "nli_score": 0.75,
        "is_hallucinated": True,
        "evidence_chunk": "The Eiffel Tower is located in Paris, France."
    }

    print("Test 1 — High confidence entailment:")
    r1 = aggregate(nli1)
    print(f"  Verdict : {r1['verdict']}")
    print(f"  LLM used: {r1['llm_used']}")
    print(f"  Explanation: {r1['explanation']}\n")

    print("Test 2 — Needs LLM check:")
    print(f"  needs_llm: {needs_llm(nli2)}")
    r2 = aggregate(nli2)
    print(f"  Verdict : {r2['verdict']}")
    print(f"  LLM used: {r2['llm_used']}")