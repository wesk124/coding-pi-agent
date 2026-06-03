from config import MAX_TOKENS, TEMPERATURE, get_model


class ModelNotAvailable(Exception):
    pass


class LLMRegistry:
    """Lazy-loading cache of llama.cpp models, keyed by discovered model_id."""

    def __init__(self):
        self._cache = {}

    def load(self, model_id: str):
        if model_id in self._cache:
            return self._cache[model_id]
        cfg = get_model(model_id)
        if cfg is None:
            raise ModelNotAvailable(
                f"Model '{model_id}' not found on disk. "
                f"Drop a .gguf file into the models directory."
            )
        from llama_cpp import Llama

        self._cache[model_id] = Llama(
            model_path=cfg["path"],
            n_threads=cfg["threads"],
            n_ctx=cfg["ctx"],
            verbose=False,
        )
        return self._cache[model_id]

    def generate(self, model_id: str, prompt: str) -> str:
        llm = self.load(model_id)
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=["### User:", "### System:"],
        )
        return output["choices"][0]["text"].strip()

    def unload(self, model_id: str):
        self._cache.pop(model_id, None)

    def loaded_ids(self) -> list:
        return list(self._cache.keys())
