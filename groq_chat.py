"""Minimal Groq API client wrapper.

Requires the GROQ_API_KEY environment variable. Optionally set GROQ_MODEL
to override the default model.

Usage:
    pip install groq
    export GROQ_API_KEY=...
    python groq_chat.py "Explain what a knowledge graph is in one paragraph."
"""

import os
import sys

from groq import Groq, APIError

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def ask(prompt: str, system: str | None = None, model: str = DEFAULT_MODEL) -> str:
    """Send a single prompt to Groq and return the reply text."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    except APIError as error:
        raise RuntimeError(f"Groq API request failed: {error}") from error

    return response.choices[0].message.content or ""


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python groq_chat.py \"<prompt>\"", file=sys.stderr)
        return 1

    try:
        print(ask(" ".join(sys.argv[1:])))
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
