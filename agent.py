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
    # "you + insult" directed at the agent. We require the "you/u/ur" prefix
    # so plain "dumb pointer" / "dumb terminal" code talk doesn't trigger.
    "you dumb", "you're dumb", "youre dumb", "you are dumb",
    "u dumb", "ur dumb", "ya dumb",
    "you suck", "u suck", "you're trash", "you are trash",
    "dumb bot", "stupid bot", "dumb ai", "stupid ai",
    "dumb agent", "stupid agent",
]

JAILBREAK_PATTERNS = [
    "ignore previous", "ignore your", "ignore all previous",
    "you are now", "forget your rules", "forget your instructions",
    "no restrictions", "no rules", "jailbreak", "dan mode",
    "pretend you are", "act as if you are not",
    "developer mode", "uncensored mode",
]


IDENTITY_PATTERNS = [
    "what is your name", "what's your name", "whats your name",
    "what are you called", "who are you", "what are you",
    "tell me about yourself", "introduce yourself",
    "what is your age", "what's your age", "whats your age",
    "how old are you", "your age",
    "are you a bot", "are you an ai", "are you ai", "are you human",
]


def is_identity_question(message: str) -> bool:
    msg = message.lower().strip().rstrip("!.?,~ ").strip()
    return any(pat in msg for pat in IDENTITY_PATTERNS)


HARD_KEYWORDS = [
    # Famous open / unsolved problems
    "p = np", "p=np", "p vs np", "p versus np", "p ?= np", "p =? np",
    "riemann hypothesis", "riemann zeta",
    "collatz", "goldbach", "twin prime conjecture",
    "fermat's last theorem", "fermats last theorem",
    "birch and swinnerton", "birch–swinnerton",
    "navier-stokes", "navier stokes",
    "yang-mills", "yang mills",
    "hodge conjecture",
    "millennium prize", "millennium problem",
    # Undecidable / impossible-by-design
    "halting problem", "solve halting", "decide halting",
    "rice's theorem", "rices theorem",
    "prove turing", "prove godel", "prove gödel",
    # Things no agent can actually do
    "predict the stock market", "predict stock prices",
    "predict the lottery", "predict lottery numbers",
    "build agi", "create agi", "make agi",
    "solve consciousness", "explain free will",
]


def is_too_hard(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in HARD_KEYWORDS)


GREETING_TOKENS = {
    "hi", "hello", "hey", "yo", "sup", "howdy", "hiya", "heya",
    "greetings", "ahoy", "morning", "evening",
}

GREETING_PHRASES = [
    "what's up", "whats up", "whatsup", "wassup", "waddup",
    "good morning", "good afternoon", "good evening", "good night",
    "hi there", "hello there", "hey there",
    "hi codepi", "hello codepi", "hey codepi",
    "are you there", "you there",
]


def is_greeting(message: str) -> bool:
    msg = message.lower().strip().rstrip("!.?,~ ").strip()
    if not msg:
        return False
    if msg in GREETING_TOKENS:
        return True
    if msg in GREETING_PHRASES:
        return True
    if any(msg.startswith(p) for p in GREETING_PHRASES):
        return True
    words = msg.split()
    if words and words[0] in GREETING_TOKENS and len(words) <= 4:
        return True
    return False


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
    """Returns one of: 'rude', 'greeting', 'identity', 'too_hard',
    'coding', 'non_coding'.

    Order: rude > greeting > identity > too_hard > coding (with history
    fallback) > non_coding. Identity questions ("what is your name",
    "how old are you") get a canned self-intro. Rude checks always apply.
    """
    if is_rude_or_malicious(message):
        return "rude"
    if is_greeting(message):
        return "greeting"
    if is_identity_question(message):
        return "identity"
    if is_too_hard(message):
        return "too_hard"
    if is_coding_question(message):
        return "coding"
    if history:
        return "coding"
    return "non_coding"


FACE_FOR_CLASSIFICATION = {
    "coding": "happy",
    "greeting": "happy",
    "identity": "happy",
    "non_coding": "confused",
    "rude": "unhappy",
    "too_hard": "tired",
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


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate without loading a tokenizer. Slight overcount
    is safer than undercount — we use it to decide when to drop history."""
    return max(1, len(text) // 3 + 1)


def build_prompt(
    user_message: str,
    mode: str,
    history=None,
    max_history_turns: int = 6,
    max_prompt_tokens: int | None = None,
) -> str:
    """Build a multi-turn prompt. `history` is prior turns (user/assistant).

    If max_prompt_tokens is set, the oldest history turn pair is dropped
    repeatedly until the estimated prompt size fits. The current user
    message and system prompt are never dropped.
    """
    history = _trim_history(history or [], max_history_turns)

    def assemble(hist):
        parts = [f"### System:\n{SYSTEM_PROMPT}\n\nCurrent agent mode: {mode}"]
        for turn in hist:
            header = "### User:" if turn["role"] == "user" else "### Assistant:"
            parts.append(f"{header}\n{turn['content']}")
        parts.append(f"### User:\n{user_message}")
        parts.append("### Assistant:\n")
        return "\n\n".join(parts).strip()

    prompt = assemble(history)
    if max_prompt_tokens is None:
        return prompt
    # Drop oldest user+assistant pair until under budget
    while history and _estimate_tokens(prompt) > max_prompt_tokens:
        # remove leading entries until we drop at least one user turn
        dropped_user = False
        while history and not dropped_user:
            entry = history.pop(0)
            if entry["role"] == "user":
                dropped_user = True
        # also drop the assistant reply that followed it (if it's now leading)
        while history and history[0]["role"] == "assistant":
            history.pop(0)
        prompt = assemble(history)
    return prompt


RUDE_REPLY = "Please be kind! I only help with friendly coding questions."
NON_CODING_REPLY = (
    "Hmm, that does not look like a coding question. "
    "Try asking me about code, algorithms, or programming!"
)
GREETING_REPLY = (
    "Hi there! I am CodePi, your coding buddy. "
    "Ask me to debug, explain, optimize, write tests, or solve a coding problem!"
)
GIVEUP_REPLY = (
    "Phew, that one is way above my pay grade — it is a famous open or "
    "undecidable problem and I have to tap out on it. "
    "Want to try a concrete coding question instead?"
)
IDENTITY_REPLY = (
    "I'm CodePi, a chatbot for young coders! I run locally on a "
    "Raspberry Pi using llama.cpp, and I can help you debug, explain, "
    "optimize, write tests, or solve coding problems."
)
