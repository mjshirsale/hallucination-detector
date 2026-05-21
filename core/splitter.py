import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

def split_into_claims(text: str) -> list[str]:
    """
    Input  : ek bada text (LLM response ya source doc)
    Output : list of individual sentences/claims
    """
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


if __name__ == "__main__":
    sample = """
    The Eiffel Tower is located in Berlin. 
    It was built in 1889. 
    The tower is 330 meters tall. 
    Napoleon Bonaparte designed it personally.
    """
    claims = split_into_claims(sample)
    print(f"Found {len(claims)} claims:")
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c}")