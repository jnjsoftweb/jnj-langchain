from dotenv import load_dotenv
import os
import requests
import json

def chat_with_perplexity(query: str) -> str:
    """
    Send a direct API request to Perplexity with debug information.
    
    Args:
        query: Question to ask
        
    Returns:
        str: Response from the API
    """
    load_dotenv()
    api_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not api_key:
        return "Error: PERPLEXITY_API_KEY not found in environment variables"
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        data = {
            "model": "llama-3.1-sonar-small-128k-online",  # Updated model name
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 1024
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data
        )
        
        print(f"\nDebug - Status Code: {response.status_code}")
        print(f"Debug - Response Headers: {dict(response.headers)}")
        
        try:
            print(f"Debug - Response Body: {response.json()}")
        except:
            print(f"Debug - Raw Response: {response.text}")
            
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    query = "Why is the Higgs Boson important?"
    print("Sending query...")
    print(f"\nQuery: {query}")
    response = chat_with_perplexity(query)
    print(f"\nFinal Response: {response}")

if __name__ == "__main__":
    main()