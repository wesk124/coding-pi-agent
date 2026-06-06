#!/usr/bin/env python3
"""Build the LeetCode RAG index from zerotrac/leetcode_problem_rating/ratings.txt.

This script is a ONE-TIME preprocessing step. It does not auto-download
anything — you download ratings.txt yourself from
https://github.com/zerotrac/leetcode_problem_rating/blob/main/ratings.txt
and pass the local path. The sentence-transformers BERT model weights are
fetched by the library on first use (HuggingFace cache).

Usage:
    python scripts/setup_rag.py path/to/ratings.txt
    python scripts/setup_rag.py path/to/ratings.txt --out-dir ./data
    python scripts/setup_rag.py path/to/ratings.txt --model bge-small-en-v1.5

Outputs (default ./data/):
    leetcode_embeddings.npy   float32, N x emb_dim
    leetcode_meta.json        list of {id, title, rating}
"""

import argparse
import json
import os
import sys


def parse_ratings(path: str) -> list:
    """Tolerant TSV parser for zerotrac ratings.txt.

    Expected columns (order can vary across snapshots):
        rating  problem_id  english_title  [chinese_title]  [contest_slug] ...

    We take col 0 as rating (must be float) and col 2 as title if present,
    falling back to the first non-numeric column.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue
            if lineno == 1 and "rating" in line.lower() and "title" in line.lower():
                continue
            cols = [c.strip() for c in line.split("\t")]
            try:
                rating = float(cols[0])
            except (ValueError, IndexError):
                continue
            pid = cols[1] if len(cols) > 1 else str(lineno)
            title = cols[2] if len(cols) > 2 else ""
            if not title:
                for c in cols[1:]:
                    if c and not c.replace(".", "").replace("-", "").isdigit():
                        title = c
                        break
            if not title:
                continue
            rows.append({"id": pid, "title": title, "rating": rating})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ratings_path", help="local path to zerotrac ratings.txt")
    ap.add_argument("--out-dir", default="./data",
                    help="output directory (default ./data)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="sentence-transformers model name")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if not os.path.isfile(args.ratings_path):
        sys.exit(f"ratings file not found: {args.ratings_path}")

    print(f"[1/3] parsing {args.ratings_path}")
    rows = parse_ratings(args.ratings_path)
    if not rows:
        sys.exit("no rows parsed — check the file format")
    print(f"      parsed {len(rows)} problems "
          f"(rating range {min(r['rating'] for r in rows):.0f} – "
          f"{max(r['rating'] for r in rows):.0f})")

    print(f"[2/3] loading BERT model: {args.model}")
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as e:
        sys.exit(
            f"missing dependency: {e}\n"
            "install with:  pip install sentence-transformers numpy"
        )
    model = SentenceTransformer(args.model)

    titles = [r["title"] for r in rows]
    print(f"[3/3] embedding {len(titles)} titles "
          f"(batch={args.batch_size})")
    emb = model.encode(
        titles,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype("float32")

    os.makedirs(args.out_dir, exist_ok=True)
    emb_path = os.path.join(args.out_dir, "leetcode_embeddings.npy")
    meta_path = os.path.join(args.out_dir, "leetcode_meta.json")
    np.save(emb_path, emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nWrote:")
    print(f"  {emb_path}   ({emb.nbytes / 1024:.1f} KB, shape {emb.shape})")
    print(f"  {meta_path}")
    print("\nRouting is enabled when you (re)start the app.")


if __name__ == "__main__":
    main()
