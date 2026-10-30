#!/usr/bin/env python3
"""
OpenAI-based Statement Classifier
=================================

Determines if a statement should be fact-checked (RESEARCH) or ignored (IGNORE).
Uses OpenAI as a gatekeeper before sending to Perplexity for actual fact-checking.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Classification prompt
CLASSIFIER_PROMPT = """You are a fact-check classifier. Given a statement, respond with ONLY "RESEARCH" or "IGNORE".

RESEARCH if the statement contains:
- Historical dates or events
- Statistics or numbers
- Scientific claims
- Verifiable facts about people, places, or things
- Claims about what someone said or did
- Claims about laws, policies, or regulations

IGNORE if the statement is:
- Opinion or preference ("I think...", "I like...", "I prefer...")
- Question (ends with ?)
- Small talk or greeting ("Hello", "How are you", "Nice to meet you")
- Subjective or unverifiable ("That's interesting", "Cool")
- Personal statements about the speaker ("I have a dog", "I'm from NYC")
- Filler words or acknowledgments ("Yeah", "Okay", "Mm-hmm", "Right")

Statement: "{text}"
Response:"""


def classify_statement(text: str, model: str = "gpt-4o-mini") -> bool:
    """
    Uses OpenAI to determine if a statement should be fact-checked.
    
    Args:
        text: The statement to classify
        model: OpenAI model to use (default: gpt-4o-mini for speed/cost)
    
    Returns:
        True if the statement should be researched (RESEARCH)
        False if it should be ignored (IGNORE)
    """
    if not text or len(text.strip()) < 5:
        return False
    
    # Force it to be a statement
    text = text.rstrip('?').strip()
    if not text.endswith('.'):
        text += '.'
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": CLASSIFIER_PROMPT.format(text=text)}
            ],
            max_tokens=10,
            temperature=0,  # Deterministic output
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result == "RESEARCH"
        
    except Exception as e:
        print(f"[Classifier Error] {e}")
        return False


def classify_statements_batch(statements: list[str], model: str = "gpt-4o-mini") -> list[dict]:
    """
    Classify multiple statements and return results with original text.
    
    Args:
        statements: List of statements to classify
        model: OpenAI model to use
    
    Returns:
        List of dicts: [{"text": "...", "should_research": True/False}, ...]
    """
    results = []
    for text in statements:
        should_research = classify_statement(text, model)
        results.append({
            "text": text,
            "should_research": should_research
        })
    return results


# ============================================================
# Test/Demo
# ============================================================

if __name__ == "__main__":
    # Test statements
    test_statements = [
        "The Eiffel Tower was built in 1889.",
        "I have two Labradoodles.",
        "Einstein failed math in school.",
        "Hello, how are you?",
        "The earth is flat.",
        "I think pizza is delicious.",
        "The population of Tokyo is over 13 million.",
        "Okay, yeah, that's interesting.",
        "The speed of light is 299,792,458 meters per second.",
        "My dog's name is Teddy.",
    ]
    
    print("=" * 60)
    print("Statement Classification Test")
    print("=" * 60)
    
    for stmt in test_statements:
        result = classify_statement(stmt)
        status = "🔍 RESEARCH" if result else "⏭️  IGNORE"
        print(f"\n{status}")
        print(f"   \"{stmt}\"")
    
    print("\n" + "=" * 60)
