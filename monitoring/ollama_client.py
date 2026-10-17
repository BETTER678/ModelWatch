import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when a local Ollama API request fails."""


def _post(endpoint, payload, timeout=300):
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    request = Request(
        f"{host}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise OllamaError(
            f"Ollama API returned HTTP {error.code}: {detail}"
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise OllamaError(f"Ollama API request failed: {error}") from error


def embed_texts(model, texts, timeout=300):
    if not texts:
        return []

    result = _post(
        "/api/embed",
        {"model": model, "input": texts, "truncate": True},
        timeout=timeout,
    )
    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise OllamaError("Ollama returned an invalid embedding batch.")
    return embeddings


def generate(model, prompt, timeout=300):
    result = _post(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    response = result.get("response")
    if not isinstance(response, str):
        raise OllamaError("Ollama returned a response without generated text.")
    return response, result
