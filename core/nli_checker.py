from transformers import pipeline

_nli = None

def get_nli_pipeline():
    global _nli
    if _nli is None:
        print("Loading NLI model... (pehli baar thoda time lagega)")
        _nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small"
        )
        print("Model loaded!")
    return _nli

def check_claim(claim: str, source: str) -> dict:
    pipe = get_nli_pipeline()
    result = pipe(
        f"{source} [SEP] {claim}",
        truncation=True,
        max_length=512
    )[0]
    
    label = result['label']   # ENTAILMENT / NEUTRAL / CONTRADICTION
    score = result['score']
    
    return {
        "claim": claim,
        "nli_label": label,
        "nli_score": round(score, 3),
        "is_hallucinated": label == "CONTRADICTION"
    }


if __name__ == "__main__":
    source = """
    The Eiffel Tower is located in Paris, France. 
    It was built in 1889 by Gustave Eiffel.
    The tower is 330 meters tall.
    """
    
    test_claims = [
        "The Eiffel Tower is located in Berlin.",      # hallucinated
        "The Eiffel Tower was built in 1889.",          # correct
        "Napoleon Bonaparte designed the Eiffel Tower." # hallucinated
    ]
    
    for claim in test_claims:
        result = check_claim(claim, source)
        status = "HALLUCINATED" if result["is_hallucinated"] else "OK"
        print(f"[{status}] {claim}")
        print(f"        Label: {result['nli_label']} | Score: {result['nli_score']}\n")