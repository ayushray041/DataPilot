"""Configuration helpers for connecting DataPilot to the Gemini SDK."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.genai import Client


# Resolve the project root from this file so loading works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL_NAME = "gemini-3.8-flash"


def get_client() -> Client:
    """Create a Gemini client using the API key from the root environment file.

    Returns:
        A configured Google GenAI client.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set or contains only whitespace.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None or not api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY must be set in the root .env file before using Gemini."
        )

    return Client(api_key=api_key)


def get_model_name() -> str:
    """Return the configured Gemini model name or the project default.

    A missing or whitespace-only GEMINI_MODEL setting uses the default model.
    """
    model_name = os.getenv("GEMINI_MODEL")
    if model_name is None or not model_name.strip():
        return DEFAULT_MODEL_NAME
    return model_name