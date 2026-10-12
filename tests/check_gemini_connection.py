
from agent.llm import get_client, get_model_name


def main():
    """Check that DataPilot can make a real Gemini API request."""

    client = get_client()
    model = get_model_name()

    print(f"Testing model: {model}")

    try:
        response = client.models.generate_content(
            model=model,
            contents=(
                "Reply with exactly: "
                "DataPilot Gemini connection successful."
            ),
        )

        if not response.text:
            print("The API responded, but returned no text.")
            return

        print("Gemini response:")
        print(response.text)
        print("\nLive API connection successful!")

    except Exception as exc:
        print(f"Gemini API request failed: {type(exc).__name__}")
        print(exc)


if __name__ == "__main__":
    main()
