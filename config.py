import os

MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.gguf")
MODEL_THREADS = int(os.getenv("MODEL_THREADS", "4"))
MODEL_CONTEXT_SIZE = int(os.getenv("MODEL_CONTEXT_SIZE", "2048"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
