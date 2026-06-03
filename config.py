import glob
import os

MODEL_THREADS = int(os.getenv("MODEL_THREADS", "4"))
MODEL_CONTEXT_SIZE = int(os.getenv("MODEL_CONTEXT_SIZE", "2048"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

MODELS_DIR = os.getenv("MODELS_DIR", "./models")
MODEL_EXTS = (".gguf",)


def _humanize(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip()


def _model_id_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.lower()


def discover_models() -> dict:
    """Scan MODELS_DIR for GGUF files and build a registry from what's there.

    Each discovered file becomes one entry. Never hardcode model names — drop a
    GGUF into the directory and it shows up automatically.
    """
    registry = {}
    if not os.path.isdir(MODELS_DIR):
        return registry
    paths = []
    for ext in MODEL_EXTS:
        paths.extend(glob.glob(os.path.join(MODELS_DIR, f"*{ext}")))
    paths.sort()
    for path in paths:
        mid = _model_id_from_path(path)
        registry[mid] = {
            "label": _humanize(os.path.splitext(os.path.basename(path))[0]),
            "path": path,
            "ctx": MODEL_CONTEXT_SIZE,
            "threads": MODEL_THREADS,
        }
    return registry


def available_model_ids() -> list:
    return list(discover_models().keys())


def get_model(model_id: str) -> dict | None:
    return discover_models().get(model_id)


def default_model_id() -> str | None:
    env_default = os.getenv("DEFAULT_MODEL_ID")
    models = discover_models()
    if env_default and env_default in models:
        return env_default
    return next(iter(models), None)
