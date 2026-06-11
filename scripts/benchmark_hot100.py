#!/usr/bin/env python3
"""LeetCode Hot-100 inference-speed benchmark across every discovered GGUF.

CLI-only. Loads each model in turn (LRU cache evicts as we go), times the
generation of 5 representative Hot-100 problems, and prints a per-model
table plus a cross-model summary. Optionally writes JSON for later analysis.

Examples:
    # benchmark every discovered model on all 5 tests
    python scripts/benchmark_hot100.py

    # cap output length to keep total runtime sane on a Pi
    python scripts/benchmark_hot100.py --max-tokens 256

    # only specific models
    python scripts/benchmark_hot100.py --models qwen2.5-coder-1.5b-instruct-q4_k_m,qwen2.5-coder-7b-instruct-q4_k_m

    # save full results for the report
    python scripts/benchmark_hot100.py --output bench_results.json
"""

import argparse
import json
import os
import statistics
import sys
import time

# Allow running from repo root or from scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import build_prompt
from config import available_model_ids, get_model
from llm import LLMRegistry, ModelNotAvailable


# 5 LeetCode Hot-100 problems spanning easy / medium / hard and covering
# different techniques (hashmap, stack, sliding window, design, two pointers).
HOT100_TESTS = [
    {
        "id": "two-sum",
        "title": "Two Sum",
        "difficulty": "easy",
        "technique": "hashmap",
        "prompt": (
            "Solve LeetCode 'Two Sum': Given an array of integers nums and an "
            "integer target, return the indices of the two numbers that add up "
            "to target. You may assume that each input has exactly one "
            "solution, and you may not use the same element twice. "
            "Provide a C++ solution with O(n) time complexity."
        ),
    },
    {
        "id": "valid-parentheses",
        "title": "Valid Parentheses",
        "difficulty": "easy",
        "technique": "stack",
        "prompt": (
            "Solve LeetCode 'Valid Parentheses': Given a string s containing "
            "just the characters '(', ')', '{', '}', '[' and ']', determine if "
            "the input string is valid. An input string is valid if open "
            "brackets are closed by the same type of brackets, and in the "
            "correct order. Provide a C++ solution."
        ),
    },
    {
        "id": "longest-substring",
        "title": "Longest Substring Without Repeating Characters",
        "difficulty": "medium",
        "technique": "sliding-window",
        "prompt": (
            "Solve LeetCode 'Longest Substring Without Repeating Characters': "
            "Given a string s, find the length of the longest substring "
            "without repeating characters. Provide a C++ solution using the "
            "sliding window technique and state the time complexity."
        ),
    },
    {
        "id": "lru-cache",
        "title": "LRU Cache",
        "difficulty": "medium",
        "technique": "design",
        "prompt": (
            "Solve LeetCode 'LRU Cache': Design a data structure that follows "
            "the constraints of a Least Recently Used (LRU) cache. Implement "
            "the LRUCache class with int capacity, int get(int key), and void "
            "put(int key, int value). All operations must run in O(1) average "
            "time. Provide a C++ solution."
        ),
    },
    {
        "id": "trapping-rain-water",
        "title": "Trapping Rain Water",
        "difficulty": "hard",
        "technique": "two-pointer",
        "prompt": (
            "Solve LeetCode 'Trapping Rain Water': Given n non-negative "
            "integers representing an elevation map where the width of each "
            "bar is 1, compute how much water it can trap after raining. "
            "Provide a C++ solution using the two-pointer approach with O(n) "
            "time and O(1) extra space, and briefly explain the invariant."
        ),
    },
]


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:5.2f}s"
    m, sec = divmod(s, 60)
    return f"{int(m):2d}m{sec:04.1f}s"


def benchmark_model(registry, model_id, tests, max_tokens):
    cfg = get_model(model_id)
    print(f"\n=== {model_id}")
    print(f"    {cfg['label']}")
    try:
        t0 = time.time()
        registry.load(model_id)
        cold = time.time() - t0
        print(f"    cold load: {cold:5.1f}s")
    except ModelNotAvailable as e:
        print(f"    ERROR: {e}")
        return None
    except Exception as e:
        print(f"    ERROR loading: {e}")
        return None

    results = []
    for t in tests:
        prompt = build_prompt(t["prompt"], "solve_problem")
        t0 = time.time()
        try:
            reply = registry.generate(model_id, prompt, max_tokens=max_tokens)
            elapsed = time.time() - t0
            tokens = max(1, len(reply.split()))
            tps = tokens / elapsed if elapsed > 0 else 0.0
            head = reply.strip().splitlines()[0][:80] if reply.strip() else "(empty)"
            print(f"    [{t['difficulty']:6s}] {t['id']:30s}  "
                  f"{fmt_seconds(elapsed)}  ~{tokens:4d} tok  ~{tps:5.2f} tok/s")
            print(f"        >> {head}")
            results.append({
                "test_id": t["id"],
                "title": t["title"],
                "difficulty": t["difficulty"],
                "technique": t["technique"],
                "elapsed_s": round(elapsed, 3),
                "approx_tokens": tokens,
                "approx_tok_per_s": round(tps, 3),
                "preview": head,
                "ok": True,
            })
        except Exception as e:
            print(f"    [{t['difficulty']:6s}] {t['id']:30s}  ERROR: {e}")
            results.append({
                "test_id": t["id"],
                "title": t["title"],
                "difficulty": t["difficulty"],
                "ok": False,
                "error": str(e),
            })

    ok = [r for r in results if r.get("ok")]
    if ok:
        mean_s = statistics.mean(r["elapsed_s"] for r in ok)
        mean_tps = statistics.mean(r["approx_tok_per_s"] for r in ok)
        total_s = sum(r["elapsed_s"] for r in ok)
        print(f"    -- mean per test: {fmt_seconds(mean_s)}  "
              f"~{mean_tps:5.2f} tok/s   "
              f"total: {fmt_seconds(total_s)}   "
              f"({len(ok)}/{len(tests)} ok)")
    return {
        "model_id": model_id,
        "label": cfg["label"],
        "cold_load_s": round(cold, 3),
        "results": results,
    }


def print_summary_table(all_results):
    rows = [m for m in all_results if m is not None]
    if not rows:
        print("\nNo successful runs.")
        return
    print("\n" + "=" * 92)
    print("SUMMARY  (mean / median across the 5 tests; cold-load shown separately)")
    print("=" * 92)
    print(f"{'model_id':50s} {'cold':>7s} {'mean':>8s} {'median':>8s} {'tok/s':>7s} {'ok':>5s}")
    print("-" * 92)
    for m in rows:
        ok = [r for r in m["results"] if r.get("ok")]
        if not ok:
            print(f"{m['model_id']:50s} {m['cold_load_s']:>6.1f}s  no successful runs")
            continue
        mean_s = statistics.mean(r["elapsed_s"] for r in ok)
        median_s = statistics.median(r["elapsed_s"] for r in ok)
        mean_tps = statistics.mean(r["approx_tok_per_s"] for r in ok)
        print(f"{m['model_id']:50s} "
              f"{m['cold_load_s']:>6.1f}s "
              f"{fmt_seconds(mean_s):>8s} "
              f"{fmt_seconds(median_s):>8s} "
              f"{mean_tps:>6.2f} "
              f"{len(ok):>2d}/{len(m['results']):<2d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models",
                    help="comma-separated model_ids (default: all discovered)")
    ap.add_argument("--output",
                    help="write full JSON results to this path")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap generated tokens per test (default: config MAX_TOKENS)")
    args = ap.parse_args()

    all_ids = available_model_ids()
    if not all_ids:
        sys.exit("No GGUFs found in models directory. Drop a *.gguf in ./models/ first.")

    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in all_ids]
        if unknown:
            sys.exit(
                f"unknown model_id(s): {', '.join(unknown)}\n"
                f"discovered: {', '.join(all_ids)}"
            )
        targets = requested
    else:
        targets = all_ids

    print(f"LeetCode Hot-100 benchmark")
    print(f"  models : {len(targets)}  ({', '.join(targets)})")
    print(f"  tests  : {len(HOT100_TESTS)}  ({', '.join(t['id'] for t in HOT100_TESTS)})")
    if args.max_tokens is not None:
        print(f"  max_tokens override: {args.max_tokens}")

    registry = LLMRegistry()
    all_results = []
    overall_t0 = time.time()
    for mid in targets:
        all_results.append(benchmark_model(registry, mid, HOT100_TESTS, args.max_tokens))
    overall = time.time() - overall_t0

    print_summary_table(all_results)
    print(f"\nTotal wall time: {fmt_seconds(overall)}")

    if args.output:
        payload = {
            "meta": {
                "max_tokens_override": args.max_tokens,
                "total_wall_seconds": round(overall, 2),
            },
            "tests": HOT100_TESTS,
            "results": all_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
