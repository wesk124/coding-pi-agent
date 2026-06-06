from collections import OrderedDict

from config import MAX_LOADED_MODELS, MAX_TOKENS, TEMPERATURE, get_model


class ModelNotAvailable(Exception):
    pass


class LLMRegistry:
    """LRU cache of llama.cpp models. Caps how many stay resident so the Pi
    doesn't OOM when the router ping-pongs between models.

    Eviction policy: when the cache is full and a new model is requested,
    the least-recently-used model is dropped from the cache (its Llama
    object is dereferenced; llama.cpp frees its weights + KV cache).
    """

    def __init__(self, max_loaded: int = MAX_LOADED_MODELS):
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._max_loaded = max(1, int(max_loaded))

    def load(self, model_id: str):
        if model_id in self._cache:
            self._cache.move_to_end(model_id)
            return self._cache[model_id]
        cfg = get_model(model_id)
        if cfg is None:
            raise ModelNotAvailable(
                f"Model '{model_id}' not found on disk. "
                f"Drop a .gguf file into the models directory."
            )
        from llama_cpp import Llama

        # Evict before loading so the new model gets the freed RAM.
        while len(self._cache) >= self._max_loaded:
            evicted_id, evicted_llm = self._cache.popitem(last=False)
            del evicted_llm  # llama.cpp frees on dereference
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
