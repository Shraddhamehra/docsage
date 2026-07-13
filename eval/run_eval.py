"""Retrieval evaluation: hit-rate@k for semantic search vs the keyword baseline.

A "hit" = the expected text appears in at least one of the top-k retrieved chunks.
Run:  python -m eval.run_eval
"""
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from api.store import keyword_search, vector_search

K = 4
EVAL_SET = Path(__file__).parent / "eval_set.json"


def hit(results: list[dict], expected: str) -> bool:
    return any(expected.lower() in r["content"].lower() for r in results)


def main() -> None:
    items = json.loads(EVAL_SET.read_text())
    vector_hits = keyword_hits = 0

    print(f"{'question':<58} vector  keyword")
    for item in items:
        q, expected = item["question"], item["expect_substring"]
        v = hit(vector_search(q, K), expected)
        kw = hit(keyword_search(q, K), expected)
        vector_hits += v
        keyword_hits += kw
        print(f"{q[:56]:<58} {'✓' if v else '✗':<7} {'✓' if kw else '✗'}")

    n = len(items)
    print(f"\nhit-rate@{K}:  vector {vector_hits}/{n} ({vector_hits/n:.0%})"
          f"  |  keyword baseline {keyword_hits}/{n} ({keyword_hits/n:.0%})")


if __name__ == "__main__":
    main()
