CODING_KEYWORDS = [
    "code", "coding", "program", "programming", "function", "class",
    "algorithm", "data structure", "bug", "debug", "compile", "compiler",
    "runtime", "syntax", "error", "leetcode", "array", "string",
    "linked list", "tree", "graph", "hashmap", "heap", "stack", "queue",
    "dynamic programming", "dp", "recursion", "binary search", "sort",
    "complexity", "big o", "test case", "unit test", "c++", "cpp",
    "python", "java", "javascript", "typescript", "sql", "rust", "go"
]


def is_coding_question(message: str) -> bool:
    msg = message.lower()
    return any(keyword in msg for keyword in CODING_KEYWORDS)


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
You are CodePi, a local coding-only AI agent running on Raspberry Pi.

Rules:
1. Only answer programming, algorithms, debugging, optimization, software engineering, and code-related questions.
2. Refuse all non-coding questions.
3. Prefer C++ unless user requests another language.
4. Keep answers concise and practical.
5. For algorithms include complexity analysis.
6. For debugging explain the likely root cause and provide a corrected version.
"""


def build_prompt(user_message: str, mode: str) -> str:
    return f"""
### System:
{SYSTEM_PROMPT}

Current agent mode: {mode}

### User:
{user_message}

### Assistant:
""".strip()
