def aggregate(nli_result: dict, llm_result: dict) -> dict:
    """
    NLI + LLM judge dono ke signals ko milata hai
    aur ek final hallucination score nikalata hai
    """
    nli_hal = 1.0 if nli_result["is_hallucinated"] else 0.0
    llm_hal = 1.0 if llm_result["llm_verdict"] == "hallucinated" else 0.0

    # NLI zyada reliable hai isliye 60% weight
    final_score = round(0.6 * nli_hal + 0.4 * llm_hal, 2)

    if final_score >= 0.6:
        verdict = "Hallucinated"
    elif final_score >= 0.3:
        verdict = "Uncertain"
    else:
        verdict = "Supported"

    return {
        "claim": nli_result["claim"],
        "hallucination_score": final_score,
        "verdict": verdict,
        "nli_label": nli_result["nli_label"],
        "llm_verdict": llm_result["llm_verdict"]
    }


if __name__ == "__main__":
    # Dummy results se test karte hain
    test_cases = [
        {
            "nli": {"claim": "Tower is in Berlin.", "nli_label": "CONTRADICTION", "nli_score": 0.94, "is_hallucinated": True},
            "llm": {"claim": "Tower is in Berlin.", "llm_verdict": "hallucinated", "llm_raw": "no"}
        },
        {
            "nli": {"claim": "Built in 1889.", "nli_label": "ENTAILMENT", "nli_score": 0.97, "is_hallucinated": False},
            "llm": {"claim": "Built in 1889.", "llm_verdict": "supported", "llm_raw": "yes"}
        },
        {
            "nli": {"claim": "Napoleon designed it.", "nli_label": "CONTRADICTION", "nli_score": 0.88, "is_hallucinated": True},
            "llm": {"claim": "Napoleon designed it.", "llm_verdict": "supported", "llm_raw": "yes"}  # LLM galat hai
        },
    ]

    for t in test_cases:
        result = aggregate(t["nli"], t["llm"])
        print(f"Claim    : {result['claim']}")
        print(f"Score    : {result['hallucination_score']}")
        print(f"Verdict  : {result['verdict']}")
        print(f"NLI      : {result['nli_label']}")
        print(f"LLM      : {result['llm_verdict']}")
        print("-" * 40)