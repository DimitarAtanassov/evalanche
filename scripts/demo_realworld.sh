#!/usr/bin/env bash
# Fetch cache-only SQuAD + AG News snapshots, materialize packs, and exercise
# the shipped evalctl surface with the mock provider.
#
# Requires: Postgres (DATABASE_URL), uv sync --extra datasets --extra dev.
# Does not commit snapshots or packs (live under .cache/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CACHE=".cache/datasets"
PACKS=".cache/packs"
OUT="reports/demo"
UA="Mozilla/5.0 (compatible; evalanche-demo/0.1)"

mkdir -p "$CACHE" "$PACKS" "$OUT"

if [[ ! -f "$CACHE/squad-dev-v1.1.json" ]]; then
  curl -fsSL -A "$UA" -o "$CACHE/squad-dev-v1.1.json" \
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
fi
if [[ ! -f "$CACHE/ag_news_test.csv" ]]; then
  curl -fsSL -A "$UA" -o "$CACHE/ag_news_test.csv" \
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
fi

python3 - <<'PY'
import hashlib
from pathlib import Path

def pin(path: Path, revision: str, url: str) -> None:
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    pin_path = path.with_name(path.name + ".pin.yaml")
    pin_path.write_text(
        f'revision: "{revision}"\n'
        f"revision_digest: {digest}\n"
        f'canonical_url: "{url}"\n',
        encoding="utf-8",
    )
    print(path.name, digest)

root = Path(".cache/datasets")
pin(
    root / "squad-dev-v1.1.json",
    "dev-v1.1",
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
)
pin(
    root / "ag_news_test.csv",
    "test-csv-v1",
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
)
PY

uv run evalctl dataset materialize --adapter squad_v1_1 \
  --source "$CACHE/squad-dev-v1.1.json" --out "$PACKS/squad-smoke" \
  --seed 42 --size 20 --tier smoke
uv run evalctl dataset materialize --adapter ag_news \
  --source "$CACHE/ag_news_test.csv" --out "$PACKS/ag-news-smoke" \
  --seed 42 --size 20 --tier smoke
uv run evalctl dataset materialize --adapter squad_v1_1 \
  --source "$CACHE/squad-dev-v1.1.json" --out "$PACKS/squad-ci" \
  --seed 7 --size 50 --tier ci
uv run evalctl dataset materialize --adapter ag_news \
  --source "$CACHE/ag_news_test.csv" --out "$PACKS/ag-news-ci" \
  --seed 7 --size 50 --tier ci

uv run evalctl dataset-validate "$PACKS/squad-smoke"
uv run evalctl dataset-validate "$PACKS/ag-news-smoke"

uv run evalctl run --dataset "$PACKS/squad-smoke" \
  --template fixtures/templates/qa_short.jinja --model mock-small --provider mock \
  --output "$OUT/squad-smoke" --concurrency 4
uv run evalctl run --dataset "$PACKS/squad-smoke" \
  --template fixtures/templates/qa_short.jinja --model mock-candidate --provider mock \
  --output "$OUT/squad-smoke-b" --concurrency 4
uv run evalctl run --dataset "$PACKS/ag-news-smoke" \
  --template fixtures/templates/classification.jinja --model mock-small --provider mock \
  --output "$OUT/ag-news-smoke" --concurrency 4
uv run evalctl run --dataset "$PACKS/squad-ci" \
  --template fixtures/templates/qa_short.jinja --model mock-small --provider mock \
  --output "$OUT/squad-ci" --concurrency 8
uv run evalctl run --dataset "$PACKS/ag-news-ci" \
  --template fixtures/templates/classification.jinja --model mock-class --provider mock \
  --output "$OUT/ag-news-ci" --concurrency 8

SQUAD_A="$(basename "$(ls "$OUT/squad-smoke"/*.json | head -1)" .json)"
SQUAD_B="$(basename "$(ls "$OUT/squad-smoke-b"/*.json | head -1)" .json)"
uv run evalctl runs rescore "$SQUAD_A" --metrics squad_f1,exact_match
uv run evalctl runs compare "$SQUAD_A" "$SQUAD_B" --metric exact_match \
  --allow-compatible --output "$OUT/compare-squad.json"

uv run evalctl power --baseline-rate 0.5 --mde 0.05 > "$OUT/power.json"
echo "Wrote artifacts under $OUT (cache-only sources stay in $CACHE)."
