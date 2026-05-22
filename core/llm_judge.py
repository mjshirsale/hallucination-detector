import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found! .env file check karo.")
        _client = Groq(api_key=api_key)
    return _client


def judge_claim(claim: str, relevant_chunks: list[str], evidence_chunk: str) -> dict:
    """
    Llama 3.3 70B se Chain-of-Thought reasoning leta hai
    Sirf tab call hota hai jab DeBERTa ne contradiction ya low confidence detect ki ho
    """
    client = get_client()

    source_context = "\n".join([f"- {chunk}" for chunk in relevant_chunks])

    prompt = f"""You are a precise fact-checking AI. Your job is to verify if a claim is supported by the source text.

SOURCE TEXT:
{source_context}

CLAIM TO VERIFY:
"{claim}"

Analyze step by step:
1. What does the source say about this topic?
2. What does the claim say?
3. Do they agree or contradict?

Respond in this EXACT format:
REASONING: [your step by step analysis in 2-3 sentences]
VERDICT: [SUPPORTED or HALLUCINATED]
CONFIDENCE: [HIGH or MEDIUM or LOW]
CORRECTION: [if HALLUCINATED, write the corrected version of the claim based on source. If SUPPORTED, write "N/A"]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()

        # Parse response
        lines = raw.split('\n')
        parsed = {}
        for line in lines:
            if line.startswith("REASONING:"):
                parsed["reasoning"] = line.replace("REASONING:", "").strip()
            elif line.startswith("VERDICT:"):
                parsed["verdict"] = line.replace("VERDICT:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                parsed["confidence"] = line.replace("CONFIDENCE:", "").strip()
            elif line.startswith("CORRECTION:"):
                parsed["correction"] = line.replace("CORRECTION:", "").strip()

        return {
            "claim": claim,
            "llm_verdict": parsed.get("verdict", "UNKNOWN"),
            "reasoning": parsed.get("reasoning", ""),
            "confidence": parsed.get("confidence", "LOW"),
            "correction": parsed.get("correction", "N/A"),
            "llm_raw": raw
        }

    except Exception as e:
        return {
            "claim": claim,
            "llm_verdict": "UNKNOWN",
            "reasoning": f"API error: {str(e)}",
            "confidence": "LOW",
            "correction": "N/A",
            "llm_raw": ""
        }


if __name__ == "__main__":
    chunks = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.",
        "The tower stands 330 meters tall. It was originally built for the 1889 World Fair.",
    ]

    test_claims = [
        "The Eiffel Tower is located in Berlin.",
        "The Eiffel Tower was built in 1889.",
        "Napoleon Bonaparte designed the Eiffel Tower.",
    ]

    for claim in test_claims:
        print(f"Claim: {claim}")
        result = judge_claim(claim, chunks, chunks[0])
        print(f"  Verdict    : {result['llm_verdict']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Reasoning  : {result['reasoning']}")
        print(f"  Correction : {result['correction']}")
        print()