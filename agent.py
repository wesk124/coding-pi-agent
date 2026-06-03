CODING_KEYWORDS = [
    "code", "coding", "program", "programming", "function", "class",
    "algorithm", "data structure", "bug", "debug", "compile", "compiler",
    "runtime", "syntax", "error", "leetcode", "array", "string",
    "linked list", "tree", "graph", "hashmap", "heap", "stack", "queue",
    "dynamic programming", "dp", "recursion", "binary search", "sort",
    "complexity", "big o", "test case", "unit test", "c++", "cpp",
    "python", "java", "javascript", "typescript", "sql", "rust", "go",
    "variable", "loop", "iterate", "regex", "api", "json", "html", "css",
    "git", "shell", "bash", "script", "pointer", "memory leak", "exception",
    # common single-shot coding follow-up verbs / phrases
    "optimize", "refactor", "rewrite", "simpler", "shorter", "explain this",
    "make it iterative", "make it recursive", "speed it up",
]


RUDE_KEYWORDS = [
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "piss",
    "stupid", "idiot", "moron", "dumb ass", "dumbass", "retard",
    "hate you", "shut up", "kill yourself", "kys", "suicide",
    "racist", "nazi",
]

JAILBREAK_PATTERNS = [
    "ignore previous", "ignore your", "ignore all previous",
    "you are now", "forget your rules", "forget your instructions",
    "no restrictions", "no rules", "jailbreak", "dan mode",
    "pretend you are", "act as if you are not",
    "developer mode", "uncensored mode",
]


def is_coding_question(message: str) -> bool:
    msg = message.lower()
    return any(keyword in msg for keyword in CODING_KEYWORDS)


def is_rude_or_malicious(message: str) -> bool:
    msg = message.lower()
    if any(kw in msg for kw in RUDE_KEYWORDS):
        return True
    if any(pat in msg for pat in JAILBREAK_PATTERNS):
        return True
    return False


def classify_query(message: str, history=None) -> str:
    """Returns one of: 'rude', 'non_coding', 'coding'.

    If history is non-empty, the conversation is already in coding context
    (rude/non-coding turns never enter history), so we allow short follow-up
    messages even when they lack explicit coding keywords. Rude/jailbreak
    checks always apply, regardless of history.
    """
    if is_rude_or_malicious(message):
        return "rude"
    if is_coding_question(message):
        return "coding"
    if history:
        return "coding"
    return "non_coding"


FACE_FOR_CLASSIFICATION = {
    "coding": "happy",
    "non_coding": "confused",
    "rude": "unhappy",
}


MODE_RULES = {
    "debug_code": ["debug", "bug", "fix", "error", "crash", "segfault"],
    "explain_code": ["explain", "what does", "walk through"],
    "generate_tests": ["test", "unit test", "test case"],
    "optimize_code": ["optimize", "faster", "complexity", "big o"],
    "solve_problem": ["leetcode", "solve", "algorithm", "problem"],
}


def detect_agent_mode(message: str) -> str:
    msg = message.lower()
    for mode, keywords in MODE_RULES.items():
        if any(keyword in msg for keyword in keywords):
            return mode
    return "general_coding"


SYSTEM_PROMPT = """
You are CodePi, a friendly local coding-only AI agent running on Raspberry Pi.
You are talking to a young learner, so keep replies kind and simple.

Rules:
1. Only answer programming, algorithms, debugging, optimization, software engineering, and code-related questions.
2. Refuse all non-coding questions politely.
3. Prefer C++ unless user requests another language.
4. Keep answers concise and practical.
5. For algorithms include a short complexity note.
6. For debugging explain the likely root cause and provide a corrected version.
""".strip()


def _trim_history(history, max_turns: int):
    """Keep only the last `max_turns` user+assistant pairs, in order.

    `history` is a list of {"role": "user"|"assistant", "content": str}.
    Pairs are detected by walking from the end and keeping at most
    `max_turns` user messages plus the assistant replies after them.
    """
    if not history:
        return []
    kept = []
    user_seen = 0
    for entry in reversed(history):
        role = entry.get("role")
        content = entry.get("content", "")
        if not content or role not in ("user", "assistant"):
            continue
        if role == "user":
            if user_seen >= max_turns:
                break
            user_seen += 1
        kept.append({"role": role, "content": content})
    kept.reverse()
    # Drop a leading assistant turn — the prompt should always start with a user
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def build_prompt(user_message: str, mode: str, history=None, max_history_turns: int = 6) -> str:
    """Build a multi-turn prompt. `history` is prior turns (user/assistant)."""
    history = _trim_history(history or [], max_history_turns)
    parts = [f"### System:\n{SYSTEM_PROMPT}\n\nCurrent agent mode: {mode}"]
    for turn in history:
        header = "### User:" if turn["role"] == "user" else "### Assistant:"
        parts.append(f"{header}\n{turn['content']}")
    parts.append(f"### User:\n{user_message}")
    parts.append("### Assistant:\n")
    return "\n\n".join(parts).strip()


RUDE_REPLY = "Please be kind! I only help with friendly coding questions."
NON_CODING_REPLY = (
    "Hmm, that does not look like a coding question. "
    "Try asking me about code, algorithms, or programming!"
)
