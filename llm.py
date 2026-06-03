from llama_cpp import Llama

from config import MODEL_CONTEXT_SIZE, MODEL_PATH, MODEL_THREADS, MAX_TOKENS, TEMPERATURE


class LocalLLM:
    def __init__(self):
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_threads=MODEL_THREADS,
            n_ctx=MODEL_CONTEXT_SIZE,
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=["### User:", "### System:"],
        )
        return output["choices"][0]["text"].strip()
