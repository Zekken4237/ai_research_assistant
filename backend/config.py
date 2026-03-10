import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILES = (
    BASE_DIR / ".env",
    BASE_DIR / "backend" / ".env",
)
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama2-uncensored:7b"


def load_env_files() -> None:
    for env_file in ENV_FILES:
        if not env_file.exists():
            continue

        for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue

            os.environ[key] = value.strip().strip("'\"")


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL


def _fetch_ollama_models() -> list[dict]:
    request = Request(
        f"{get_ollama_base_url()}/api/tags",
        headers={"Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=5) as response:
            payload = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(
            f'Ollama is not reachable at {get_ollama_base_url()}. Start Ollama and try again.'
        ) from exc

    data = json.loads(payload)
    models = data.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Ollama returned an invalid model list.")

    return models


def _resolve_model_name(preferred_model: str, models: list[dict]) -> str | None:
    available_names = [
        model.get("name", "").strip()
        for model in models
        if model.get("name")
    ]

    if preferred_model in available_names:
        return preferred_model

    preferred_lower = preferred_model.lower()

    for name in available_names:
        if name.lower() == preferred_lower:
            return name

    for name in available_names:
        if preferred_lower in name.lower():
            return name

    return None


def get_ollama_status() -> dict:
    preferred_model = get_ollama_model()
    base_url = get_ollama_base_url()

    try:
        models = _fetch_ollama_models()
    except RuntimeError as exc:
        return {
            "available": False,
            "base_url": base_url,
            "model": preferred_model,
            "resolved_model": None,
            "detail": str(exc),
        }

    resolved_model = _resolve_model_name(preferred_model, models)
    if resolved_model:
        return {
            "available": True,
            "base_url": base_url,
            "model": preferred_model,
            "resolved_model": resolved_model,
            "detail": None,
        }

    available_models = [
        model.get("name", "").strip()
        for model in models
        if model.get("name")
    ]
    detail = (
        f'Ollama model "{preferred_model}" is not available. '
        f'Available models: {", ".join(available_models) or "none"}.'
    )
    return {
        "available": False,
        "base_url": base_url,
        "model": preferred_model,
        "resolved_model": None,
        "detail": detail,
    }


def require_ollama_ready() -> str:
    status = get_ollama_status()
    if status["available"]:
        return status["resolved_model"]

    raise RuntimeError(status["detail"])
