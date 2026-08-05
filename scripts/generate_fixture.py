#!/usr/bin/env python3
"""Generate evaluation dataset fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evalharness.hashing import sha256_hex


def generate_cases(n: int) -> list[dict]:
    cases = []
    for i in range(n):
        cases.append(
            {
                "id": f"case-{i:05d}",
                "task_type": "qa_short",
                "inputs": {"question": f"What is {i} plus one?"},
                "reference_answer": str(i + 1),
                "slices": {"difficulty": "easy" if i % 2 == 0 else "hard", "lang": "en"},
                "weight": 1.0,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("fixtures/large_dataset"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cases = generate_cases(args.cases)
    lines = [json.dumps(c, ensure_ascii=False) for c in cases]
    content_sha256 = sha256_hex("\n".join(lines).encode("utf-8"))

    manifest = {
        "name": "synthetic-qa",
        "version": "1.0.0",
        "split": "dev",
        "license": "CC0-1.0",
        "pii_scrubbed": True,
        "created_at": "2026-08-05",
        "slices": ["difficulty", "lang"],
        "content_sha256": content_sha256,
    }
    (args.output / "manifest.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (args.output / "cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.cases} cases to {args.output} (sha256={content_sha256})")


if __name__ == "__main__":
    main()
