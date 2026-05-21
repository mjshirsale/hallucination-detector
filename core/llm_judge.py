from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None:
        print("Loading Flan-T5 model... (pehli baar thoda time lagega)")
        _tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        _model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        print("Model loaded!")

def judge_claim(claim: str, source: str) -> dict:
    load_model()
    prompt = (
        f"Does the following source support the claim?\n"
        f"Source: {source[:500]}\n"
        f"Claim: {claim}\n"
        f"Answer yes or no:"
    )
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=5)
    
    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    supported = "yes" in answer
    
    return {
        "claim": claim,
        "llm_verdict": "supported" if supported else "hallucinated",
        "llm_raw": answer
    }


if __name__ == "__main__":
    source = """
    The Eiffel Tower is located in Paris, France.
    It was built in 1889 by Gustave Eiffel.
    The tower is 330 meters tall.
    """

    test_claims = [
        "The Eiffel Tower is located in Berlin.",
        "The Eiffel Tower was built in 1889.",
        "Napoleon Bonaparte designed the Eiffel Tower."
    ]

    for claim in test_claims:
        result = judge_claim(claim, source)
        status = "HALLUCINATED" if result["llm_verdict"] == "hallucinated" else "OK"
        print(f"[{status}] {claim}")
        print(f"        LLM answer: {result['llm_raw']}\n")