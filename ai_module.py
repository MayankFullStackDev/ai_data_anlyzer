import google.generativeai as genai
from utils import generate_data_context_prompt
import pandas as pd
import os
from dotenv import load_dotenv
from google.api_core.exceptions import InvalidArgument, NotFound, GoogleAPIError

# Load .env file
load_dotenv()

# Verify API key loading with a debug print (without exposing the full key)
env_api_key = os.getenv("GEMINI_API_KEY")
if env_api_key:
    print(f"[DEBUG] Loaded GEMINI_API_KEY from environment: {env_api_key[:5]}...")
else:
    print("[DEBUG] GEMINI_API_KEY environment variable is not set.")


def validate_api_key(api_key: str) -> bool:
    """
    Validates the provided Gemini API key using a lightweight test request.
    Handles exceptions properly to distinguish between invalid keys,
    missing models, and network/API errors.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[DEBUG] validate_api_key failed: No API key was provided or found in the environment.")
            return False

    print(f"[DEBUG] Validating API key: {api_key[:5]}...")

    try:
        # Configure the SDK with the key
        genai.configure(api_key=api_key)

        # Initialize the model (using the standard gemini-1.5-flash-latest model as primary)
        model_name = "gemini-1.5-flash-latest"
        model = genai.GenerativeModel(model_name)

        # Lightweight test request with safety fallback if gemini-1.5-flash-latest is not available
        try:
            response = model.generate_content(
                "Hello",
                generation_config={"max_output_tokens": 5}
            )
            return response.text is not None
        except NotFound:
            print(f"[DEBUG] Model '{model_name}' not found. Trying fallback 'gemini-flash-latest'...")
            model = genai.GenerativeModel("gemini-flash-latest")
            response = model.generate_content(
                "Hello",
                generation_config={"max_output_tokens": 5}
            )
            return response.text is not None

    except InvalidArgument as e:
        print(f"[ERROR] API Key Validation Error: Invalid API key. (Details: {e})")
        return False
    except NotFound as e:
        print(f"[ERROR] API Key Validation Error: Fallback Gemini model was also not found. (Details: {e})")
        return False
    except GoogleAPIError as e:
        print(f"[ERROR] API Key Validation Error: Network or Gemini API error. (Details: {e})")
        return False
    except Exception as e:
        print(f"[ERROR] API Key Validation Error: An unexpected error occurred: {e}")
        return False


def generate_automated_insights(df: pd.DataFrame, api_key: str) -> str:
    """
    Generates automated data insights (Trends, Patterns, Anomalies) using Gemini.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[ERROR] generate_automated_insights: Gemini API key is not configured.")
            return "Error: Gemini API key is missing or not configured. Please provide a valid API key in the sidebar."

    print(f"[DEBUG] generate_automated_insights using API key: {api_key[:5]}...")

    try:
        context_prompt = generate_data_context_prompt(df)

        genai.configure(api_key=api_key)

        system_instruction = """
        You are an expert Data Scientist.

        Give output in 3 sections:
        1. Trends
        2. Patterns
        3. Anomalies

        Be specific and use column names.
        """

        model_name = "gemini-1.5-flash-latest"
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )

        prompt = f"Analyze this dataset:\n\n{context_prompt}"

        # Execute request with fallback if gemini-1.5-flash-latest is not available
        try:
            response = model.generate_content(prompt)
        except NotFound:
            print(f"[DEBUG] Model '{model_name}' not found. Trying fallback 'gemini-flash-latest'...")
            model = genai.GenerativeModel(
                model_name="gemini-flash-latest",
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)

        return response.text if response.text else "No response from AI."

    except InvalidArgument as e:
        print(f"[ERROR] generate_automated_insights: Invalid API Key. (Details: {e})")
        return f"Error: Invalid Gemini API key. Please check your credentials. (Details: {e})"
    except NotFound as e:
        print(f"[ERROR] generate_automated_insights: Fallback model not found. (Details: {e})")
        return f"Error: The requested Gemini models were not found. (Details: {e})"
    except GoogleAPIError as e:
        print(f"[ERROR] generate_automated_insights: Network or Gemini API error. (Details: {e})")
        return f"Error: Gemini API or Network error. Please try again. (Details: {e})"
    except Exception as e:
        print(f"[ERROR] generate_automated_insights: Unexpected error: {e}")
        return f"Error: An unexpected error occurred: {str(e)}"


def chat_with_dataset(df: pd.DataFrame, user_query: str, chat_history: list, api_key: str) -> str:
    """
    Facilitates natural language chat with the dataset summary context using Gemini.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[ERROR] chat_with_dataset: Gemini API key is not configured.")
            return "Chat error: Gemini API key is missing or not configured. Please provide a valid API key in the sidebar."

    print(f"[DEBUG] chat_with_dataset using API key: {api_key[:5]}...")

    try:
        context_prompt = generate_data_context_prompt(df)

        genai.configure(api_key=api_key)

        system_instruction = f"""
        You are an AI Data Assistant.

        Dataset summary:
        {context_prompt}

        Answer user queries clearly using dataset.
        """

        model_name = "gemini-1.5-flash-latest"
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )

        # Convert chat history to the format expected by Gemini SDK
        gemini_history = []
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        # Execute chat with fallback if gemini-1.5-flash-latest is not available
        try:
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_query)
        except NotFound:
            print(f"[DEBUG] Model '{model_name}' not found. Trying fallback 'gemini-flash-latest'...")
            model = genai.GenerativeModel(
                model_name="gemini-flash-latest",
                system_instruction=system_instruction
            )
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_query)

        return response.text if response.text else "No response."

    except InvalidArgument as e:
        print(f"[ERROR] chat_with_dataset: Invalid API Key. (Details: {e})")
        return f"Chat error: Invalid Gemini API key. Please check your credentials. (Details: {e})"
    except NotFound as e:
        print(f"[ERROR] chat_with_dataset: Fallback model not found. (Details: {e})")
        return f"Chat error: The requested Gemini models were not found. (Details: {e})"
    except GoogleAPIError as e:
        print(f"[ERROR] chat_with_dataset: Network or Gemini API error. (Details: {e})")
        return f"Chat error: Gemini API or Network error. Please try again. (Details: {e})"
    except Exception as e:
        print(f"[ERROR] chat_with_dataset: Unexpected error: {e}")
        return f"Chat error: {str(e)}"