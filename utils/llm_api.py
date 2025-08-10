# utils/llm_api.py
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
APIPE_API_KEY = os.getenv("APIPE_API_KEY")

def call_llm_api(messages):
    """
    Calls the LLM API to get a response.
    """
    if not APIPE_API_KEY:
        return {"error": "APIPE_API_KEY is not set in the environment."}

    api_url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {APIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": messages,
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=90)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = f"API request failed: {e}. Response: {response.text if 'response' in locals() else 'No response'}"
        print(error_message)
        return {"error": error_message}
