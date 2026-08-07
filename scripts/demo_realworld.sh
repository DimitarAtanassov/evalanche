#!/usr/bin/env bash
# Fetch cache-only SQuAD + AG News snapshots, materialize packs, and exercise
# evalctl end-to-end (default: Ollama llama3.2:1b).
#
# Requires: Postgres (DATABASE_URL), uv sync --extra datasets --extra dev.
# For Ollama: docker compose up -d ollama && ollama pull <model>
# Does not commit snapshots or packs (live under .cache/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROVIDER="${PROVIDER:-ollama}"
MODEL="${MODEL:-llama3.2:1b}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

CACHE=".cache/datasets"
PACKS=".cache/packs"
OUT="reports/demo/${PROVIDER}"
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

uv run evalctl dataset-validate "$PACKS/squad-smoke"
uv run evalctl dataset-validate "$PACKS/ag-news-smoke"

run() {
  local name="$1" ds="$2" tpl="$3"
  shift 3
  uv run evalctl run --dataset "$ds" --template "$tpl" \
    --model "$MODEL" --provider "$PROVIDER" \
    --output "$OUT/$name" --concurrency 2 --temperature 0.0 --seed 42 "$@"
}

run squad-smoke "$PACKS/squad-smoke" fixtures/templates/qa_short.jinja --max-tokens 64
run squad-smoke-b "$PACKS/squad-smoke" fixtures/templates/qa_short.jinja --max-tokens 64 --temperature 0.2 --seed 7
run ag-news-smoke "$PACKS/ag-news-smoke" fixtures/templates/classification.jinja --max-tokens 32
run synth-finance fixtures/datasets/synthetic-finance-smoke fixtures/templates/numeric.jinja --max-tokens 32
run synth-math fixtures/datasets/synthetic-math-smoke fixtures/templates/numeric.jinja --max-tokens 32
run synth-news fixtures/datasets/synthetic-news-smoke fixtures/templates/classification.jinja --max-tokens 32
run synth-summarization fixtures/datasets/synthetic-summarization-smoke fixtures/templates/summarization.jinja --max-tokens 128
run synth-extraction fixtures/datasets/synthetic-extraction-smoke fixtures/templates/extraction.jinja --max-tokens 128

SQUAD_A="$(basename "$(ls "$OUT/squad-smoke"/*.json | head -1)" .json)"
SQUAD_B="$(basename "$(ls "$OUT/squad-smoke-b"/*.json | head -1)" .json)"
uv run evalctl runs rescore "$SQUAD_A" --metrics squad_f1,exact_match
uv run evalctl runs compare "$SQUAD_A" "$SQUAD_B" --metric exact_match \
  --allow-compatible --output "$OUT/compare-squad.json"

echo "Wrote artifacts under $OUT (provider=$PROVIDER model=$MODEL)."
