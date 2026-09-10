#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Ollama client using the HTTP API.
If the API is not available, it uses the Ollama CLI.
"""

from __future__ import annotations
import json
import shutil
import subprocess
import urllib.request
import urllib.error
from typing import List, Generator

DEFAULT_MODEL = "llama3:8b"
OLLAMA_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Error while using Ollama."""


def is_ollama_available() -> bool:
    """Check if the Ollama server is running or the CLI is available."""
    try:
        # Try the local API
        with urllib.request.urlopen(OLLAMA_URL, timeout=1.5) as response:
            if response.status == 200:
                return True
    except Exception:
        pass

    # Check if the CLI exists
    return shutil.which("ollama") is not None


def list_models() -> List[str]:
    """Return a list of installed models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
            return [model["name"] for model in data.get("models", [])]
    except Exception:
        # Use the CLI if the API is not available
        if shutil.which("ollama") is not None:
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().splitlines()
                    if len(lines) > 1:
                        return [line.split()[0] for line in lines[1:] if line.strip()]
            except Exception:
                pass
        return []


def run_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> Generator[str, None, None]:
    """Send a prompt to Ollama and return generated tokens.

    If the API is not available, the CLI is used instead.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        # Read streamed tokens from the API
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for line in response:
                if line:
                    res_data = json.loads(line.decode("utf-8"))
                    token = res_data.get("response", "")
                    if token:
                        yield token
                    if res_data.get("done", False):
                        break

    except urllib.error.URLError as e:
        # Use the CLI if the API is not available
        if not shutil.which("ollama"):
            raise OllamaError(
                f"Не вдалося з'єднатися з Ollama API ({e.reason if hasattr(e, 'reason') else e}).\n"
                "Команду 'ollama' також не знайдено в PATH для резервного запуску."
            )

        try:
            # Run the CLI and get the full answer
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise OllamaError(f"Ollama CLI не відповіла за {timeout}с (модель '{model}').")
        except OSError as cli_err:
            raise OllamaError(f"Не вдалося запустити ollama CLI: {cli_err}")

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise OllamaError(
                f"Ollama API недоступне ({e.reason if hasattr(e, 'reason') else e}).\n"
                f"Спроба запустити через CLI повернула помилку: {stderr}"
            )

        # Return the full CLI output
        yield result.stdout.strip()

    except Exception as e:
        raise OllamaError(f"Помилка при генерації відповіді Ollama: {e}")


GROUNDED_PROMPT_TEMPLATE = """You are a grounded assistant.

Use ONLY the context below to answer the question.

If the answer is absent from the context, answer exactly:
"I don't know based on the provided documentation."

------------------------
Context:
{context}
------------------------
Question:
{question}
------------------------
Answer:"""


def build_grounded_prompt(question: str, context: str) -> str:
    return GROUNDED_PROMPT_TEMPLATE.format(context=context, question=question)


def ask_ollama_grounded(question: str, context_chunks: List[str], model: str = DEFAULT_MODEL) -> Generator[str, None, None]:
    """Use Ollama with RAG context."""
    context = "\n\n".join(context_chunks)
    prompt = build_grounded_prompt(question, context)
    yield from run_ollama(prompt, model=model)


def ask_ollama_freeform(question: str, model: str = DEFAULT_MODEL) -> Generator[str, None, None]:
    """Use Ollama without RAG."""
    yield from run_ollama(question, model=model)