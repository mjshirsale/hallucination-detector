import torch
from transformers import pipeline

_nli = None

def get_nli_pipeline():
    global _nli
    if _nli is None:
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            print(f"GPU detected: {torch.cuda.get_device_name(0)} — loading DeBERTa-large on GPU!")
        else:
            print("GPU not found — loading on CPU (slower)")

        _nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-large",
            device=device,
            max_length=512,
            truncation=True
        )
        print("NLI model ready!\n")
    return _nli


def check_claim(claim: str, relevant_chunks: list[str]) -> dict:
    """
    Claim ko top retrieved chunks ke against check karta hai
    Har chunk pe NLI run karta hai, worst case return karta hai
    """
    pipe = get_nli_pipeline()

    best_result = None
    best_score = -1

    for chunk in relevant_chunks:
        inputs = f"{chunk} [SEP] {claim}"
        result = pipe(inputs)[0]
        label = result['label']
        score = result['score']

        # Contradiction ya highest confidence wala chunk rakho
        if label == "CONTRADICTION":
            if score > best_score:
                best_score = score
                best_result = result
                best_chunk = chunk
        elif best_result is None:
            best_result = result
            best_chunk = chunk
            best_score = score

    label = best_result['label']
    score = best_result['score']

    return {
        "claim": claim,
        "nli_label": label,
        "nli_score": round(score, 3),
        "is_hallucinated": label.upper() == "CONTRADICTION",
        "evidence_chunk": best_chunk
    }


if __name__ == "__main__":
    source_chunks = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.",
        "The tower stands 330 meters tall. It was originally built for the 1889 World Fair.",
    ]

    test_claims = [
        "The Eiffel Tower is located in Berlin.",
        "The Eiffel Tower was built in 1889.",
        "Napoleon Bonaparte designed the Eiffel Tower.",
    ]

    for claim in test_claims:
        result = check_claim(claim, source_chunks)
        status = "HALLUCINATED" if result["is_hallucinated"] else "OK"
        print(f"[{status}] {claim}")
        print(f"  Label    : {result['nli_label']}")
        print(f"  Score    : {result['nli_score']}")
        print(f"  Evidence : {result['evidence_chunk'][:60]}...")
        print()