"""Configuration helpers for connecting DataPilot to the Gemini SDK."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.genai import Client
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


# Resolve the project root from this file so loading works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL_NAME = "gemini-3.8-flash"


def _get_setting(name: str) -> str | None:
    """Return a non-blank setting from the environment or Streamlit secrets."""
    value = os.getenv(name)
    if value is not None and value.strip():
        return value

    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        return None

    if isinstance(value, str) and value.strip():
        return value
    return None


def get_client() -> Client:
    """Create a Gemini client using the configured API key.

    The key is read from the environment (including the root ``.env`` file),
    then from Streamlit secrets when it is not configured there.

    Raises:
        RuntimeError: If no non-blank GEMINI_API_KEY is configured.
    """
    api_key = _get_setting("GEMINI_API_KEY")
    if api_key is None:
        raise RuntimeError(
            "GEMINI_API_KEY must be configured in the environment, root .env "
            "file, or Streamlit secrets before using Gemini."
        )

    return Client(api_key=api_key)


def get_model_name() -> str:
    """Return the configured Gemini model name or the existing project default.

    The environment is checked first, followed by Streamlit secrets. Missing
    or blank values use ``DEFAULT_MODEL_NAME``.
    """
    return _get_setting("GEMINI_MODEL") or DEFAULT_MODEL_NAME