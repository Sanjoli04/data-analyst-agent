# utils/llm_api.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

# Configure the generative AI client
if GOOGLE_GEMINI_API_KEY:
    genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

def call_llm_api(messages):
    """
    Calls the Google Gemini API to get a response, with a timeout.
    """
    if not GOOGLE_GEMINI_API_KEY:
        return {"error": "GOOGLE_GEMINI_API_KEY is not set in the environment."}

    try:
        # Ensure the model is available. This is a lightweight check.
        if not genai.get_model("models/gemini-1.5-flash-latest"):
             return {"error": "Gemini 1.5 Flash model is not available."}
    except Exception as e:
        return {"error": f"Failed to get model info from Gemini API: {e}"}


    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # The Gemini API expects a list of content parts, not a direct message list.
    # We will take the content from the last user message for the prompt.
    prompt = ""
    if messages and isinstance(messages, list) and len(messages) > 0:
        last_message = messages[-1]
        if isinstance(last_message, dict) and last_message.get("role") == "user":
            prompt = last_message.get("content", "")

    if not prompt:
        return {"error": "Could not extract a valid prompt from the messages list."}

    try:
        # --- FIX: Added a 60-second timeout to the API request ---
        # This prevents the server from hanging indefinitely if the API is slow to respond.
        response = model.generate_content(
            prompt,
            request_options={"timeout": 60}
        )
        
        # Construct a response that mimics the old structure for compatibility
        return {
            "choices": [
                {
                    "message": {
                        "content": response.text
                    }
                }
            ]
        }
    except Exception as e:
        # This will now catch timeout errors as well as other API failures.
        error_message = f"Gemini API request failed: {e}"
        print(error_message)
        return {"error": error_message}
