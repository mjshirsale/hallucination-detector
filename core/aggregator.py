def aggregate(nli_result: dict, llm_result: dict = None,
              fact_result: dict = None) -> dict:
    """
    3-signal aggregator:
    NLI (40%) + LLM Judge (35%) + Fact Checker (25%)
    """
    nli_label = nli_result["nli_label"].upper()
    nli_score = nli_result["nli_score"]
    claim = nli_result["claim"]

    fact_score = 0.0
    fact_signals = []
    if fact_result:
        fact_score = fact_result["contradiction_score"]
        fact_signals = fact_result.get("signals", [])

    # --- Case 1: Fact checker strong contradiction ---
    if fact_score >= 0.5:
        llm_hal = (llm_result and
                   llm_result["llm_verdict"] == "HALLUCINATED")

        final_score = min(
            0.5 * fact_score +
            0.3 * (1.0 if nli_label == "CONTRADICTION" else 0.0) +
            0.2 * (1.0 if llm_hal else 0.0),
            1.0
        )

        if final_score >= 0.4:
            explanation = f"Fact check failed: {'; '.join(fact_signals[:2])}"
            if llm_result and llm_result.get("reasoning"):
                explanation += f" | {llm_result['reasoning']}"

            return {
                "claim": claim,
                "verdict": "Hallucinated",
                "hallucination_score": round(final_score, 2),
                "confidence": "HIGH" if final_score >= 0.6 else "MEDIUM",
                "explanation": explanation,
                "correction": llm_result.get("correction", "N/A") if llm_result else "N/A",
                "reasoning": llm_result.get("reasoning", "") if llm_result else "",
                "evidence": nli_result.get("evidence_chunk", ""),
                "fact_signals": fact_signals,
                "llm_used": llm_result is not None
            }

    # --- Case 2: NLI confident entailment + no fact issues ---
    if nli_label == "ENTAILMENT" and nli_score >= 0.85 and fact_score < 0.2:
        return {
            "claim": claim,
            "verdict": "Supported",
            "hallucination_score": 0.0,
            "confidence": "HIGH",
            "explanation": "Source directly supports this claim.",
            "correction": "N/A",
            "reasoning": "",
            "evidence": nli_result.get("evidence_chunk", ""),
            "fact_signals": [],
            "llm_used": False
        }

    # --- Case 3: NLI confident contradiction ---
    if nli_label == "CONTRADICTION" and nli_score >= 0.85:
        final_score = min(
            0.6 + (0.2 * fact_score) +
            (0.2 if llm_result and
             llm_result["llm_verdict"] == "HALLUCINATED" else 0.0),
            1.0
        )
        explanation = "Source contradicts this claim."
        if fact_signals:
            explanation += f" | {'; '.join(fact_signals[:2])}"
        if llm_result and llm_result.get("reasoning"):
            explanation += f" | {llm_result['reasoning']}"

        return {
            "claim": claim,
            "verdict": "Hallucinated",
            "hallucination_score": round(final_score, 2),
            "confidence": "HIGH",
            "explanation": explanation,
            "correction": llm_result.get("correction", "N/A") if llm_result else "N/A",
            "reasoning": llm_result.get("reasoning", "") if llm_result else "",
            "evidence": nli_result.get("evidence_chunk", ""),
            "fact_signals": fact_signals,
            "llm_used": llm_result is not None
        }

    # --- Case 4: LLM result available ---
    if llm_result:
        nli_hal = 1.0 if nli_label == "CONTRADICTION" else 0.0
        llm_hal = 1.0 if llm_result["llm_verdict"] == "HALLUCINATED" else 0.0

        final_score = round(
            0.4 * nli_hal +
            0.35 * llm_hal +
            0.25 * fact_score,
            2
        )

        if final_score >= 0.6:
            verdict = "Hallucinated"
        elif final_score >= 0.3:
            verdict = "Uncertain"
        else:
            verdict = "Supported"

        conflict = (nli_label == "CONTRADICTION") != (llm_hal == 1.0)
        explanation = llm_result.get("reasoning", "")
        if conflict:
            explanation = f"[Models disagree] {explanation}"
        if fact_signals:
            explanation += f" | Fact issues: {'; '.join(fact_signals[:2])}"

        return {
            "claim": claim,
            "verdict": verdict,
            "hallucination_score": final_score,
            "confidence": llm_result.get("confidence", "MEDIUM"),
            "explanation": explanation,
            "correction": llm_result.get("correction", "N/A"),
            "reasoning": llm_result.get("reasoning", ""),
            "evidence": nli_result.get("evidence_chunk", ""),
            "fact_signals": fact_signals,
            "llm_used": True
        }

    # --- Case 5: Uncertain ---
    return {
        "claim": claim,
        "verdict": "Uncertain",
        "hallucination_score": round(0.3 + (0.2 * fact_score), 2),
        "confidence": "LOW",
        "explanation": f"Insufficient evidence. {'; '.join(fact_signals[:2]) if fact_signals else ''}",
        "correction": "N/A",
        "reasoning": "",
        "evidence": nli_result.get("evidence_chunk", ""),
        "fact_signals": fact_signals,
        "llm_used": False
    }


def needs_llm(nli_result: dict, fact_result: dict = None) -> bool:
    """Smart filter — LLM call karni chahiye?"""
    label = nli_result["nli_label"].upper()
    score = nli_result["nli_score"]
    fact_score = fact_result["contradiction_score"] if fact_result else 0.0

    # Fact checker strong signal — LLM se confirm karo
    if fact_score >= 0.4:
        return True

    # NLI confident entailment + no fact issues — skip
    if label == "ENTAILMENT" and score >= 0.85 and fact_score < 0.2:
        return False

    # NLI very confident contradiction — skip
    if label == "CONTRADICTION" and score >= 0.92:
        return False

    return True