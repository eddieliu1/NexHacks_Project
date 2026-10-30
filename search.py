import os
import requests
from pprint import pprint
from dotenv import load_dotenv

# ============================================================
# Perplexity Sonar API - Quick Search & Link Extraction
# ============================================================
# 
# Setup:
# 1. Get your API key from https://www.perplexity.ai/settings/api
# 2. Create a .env file with: PPLX_API_KEY="your-api-key-here"
#
# ============================================================

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("PPLX_API_KEY")
BASE_URL = "https://api.perplexity.ai/chat/completions"


def search(query: str, model: str = "sonar") -> dict:
    """
    Run a search query using Perplexity Sonar API.
    
    Args:
        query: Your search query
        model: "sonar" (fast/cheap) or "sonar-pro" (thorough/more sources)
    
    Returns:
        Full API response as dict
    """
    if not API_KEY:
        raise ValueError(
            "Missing API key! Set PPLX_API_KEY environment variable.\n"
            "Get your key at: https://www.perplexity.ai/settings/api"
        )
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query}
        ],
        "stream": False
    }
    
    response = requests.post(BASE_URL, json=body, headers=headers)
    response.raise_for_status()
    return response.json()


def extract_links(response: dict) -> list[dict]:
    """
    Extract links from the API response.
    
    Returns a list of dicts with keys: title, url, snippet
    """
    results = []
    
    # Primary location for search results
    if "search_results" in response:
        for item in response["search_results"]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", "")
            })
    
    # Fallback: citations field (older responses)
    elif "citations" in response:
        for url in response["citations"]:
            results.append({
                "title": "",
                "url": url,
                "snippet": ""
            })
    
    return results

def quick_search(query: str, model: str = "sonar") -> None:
    """
    Run a search and print the answer + extracted links nicely.
    """
    print(f"\n🔍 Searching: {query}\n")
    print("-" * 60)
    
    data = search(query, model)
    
    # Print the answer
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("\n📝 Answer:\n")
    print(answer)
    
    # Print extracted links
    links = extract_links(data)
    if links:
        print("\n\n🔗 Sources:\n")
        for i, link in enumerate(links, 1):
            print(f"{i}. {link['title'] or 'Untitled'}")
            print(f"   {link['url']}")
            if link['snippet']:
                print(f"   → {link['snippet'][:100]}...")
            print()
    
    print("-" * 60)

# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Better test query
    query = "The Eiffel Tower was completed in 1889 for the World's Fair."
    quick_search(query)
    
    # Or get raw data for custom processing:
    # response = search("your query here")
    # links = extract_links(response)
    # pprint(links)