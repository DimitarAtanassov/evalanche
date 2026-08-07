#!/usr/bin/env bash
# Fetch real public snapshots (SQuAD, AG News, FinQA, PubMedQA), materialize
# packs, and run evalctl end-to-end (default: Ollama llama3.2:1b).
#
# Requires: Postgres, uv sync --extra datasets --extra dev.
# For Ollama: docker compose up -d ollama && ollama pull <model>
# Cache-only: snapshots under .cache/ (gitignored).
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
OUT="reports/demo/${PROVIDER}-real"
UA="Mozilla/5.0 (compatible; evalanche-demo/0.1)"
mkdir -p "$CACHE" "$PACKS" "$OUT"

fetch() {
  local url="$1" dest="$2"
  if [[ ! -f "$dest" ]]; then
    curl -fsSL -A "$UA" -o "$dest" "$url"
  fi
}

fetch "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" \
  "$CACHE/squad-dev-v1.1.json"
fetch "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv" \
  "$CACHE/ag_news_test.csv"
fetch "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json" \
  "$CACHE/finqa-train.json"
fetch "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json" \
  "$CACHE/pubmedqa-labeled.json"

python3 - <<'PY'
import hashlib, json
from pathlib import Path

def pin(path: Path, revision: str, url: str) -> None:
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".pin.yaml").write_text(
        f'revision: "{revision}"\nrevision_digest: {digest}\ncanonical_url: "{url}"\n',
        encoding="utf-8",
    )
    print(path.name, digest)

root = Path(".cache/datasets")
pin(root / "squad-dev-v1.1.json", "dev-v1.1",
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
pin(root / "ag_news_test.csv", "test-csv-v1",
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv")
pin(root / "pubmedqa-labeled.json", "ori_pqal-v1",
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json")

# FinQA: keep only answers that fit pack field bounds.
src = root / "finqa-train.json"
data = json.loads(src.read_text(encoding="utf-8"))
kept = []
for item in data:
    qa = item.get("qa") or {}
    ans = str(qa.get("answer", "")).strip()
    q = str(qa.get("question", "")).strip()
    if not ans or not q:
        continue
    ctx = "\n".join(map(str, [*item.get("pre_text", []), *item.get("post_text", [])]))
    if len(ctx) > 2000 or len(q) > 2000 or len(ans) > 500:
        continue
    kept.append(item)
bounded = root / "finqa-train-bounded.json"
bounded.write_text(json.dumps(kept), encoding="utf-8")
pin(bounded, "train-bounded-v1",
    "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json")
print("finqa kept", len(kept))
PY

mat() {
  local adapter="$1" source="$2" out="$3" size="$4" tier="$5" seed="$6"
  rm -rf "$out"
  uv run evalctl dataset materialize --adapter "$adapter" --source "$source" \
    --out "$out" --seed "$seed" --size "$size" --tier "$tier"
}

mat squad_v1_1 "$CACHE/squad-dev-v1.1.json" "$PACKS/squad-ci" 50 ci 7
mat ag_news "$CACHE/ag_news_test.csv" "$PACKS/ag-news-ci" 50 ci 7
mat finqa "$CACHE/finqa-train-bounded.json" "$PACKS/finqa-smoke" 20 smoke 42
mat pubmedqa "$CACHE/pubmedqa-labeled.json" "$PACKS/pubmedqa-smoke" 20 smoke 42

run() {
  local name="$1" ds="$2" tpl="$3"
  shift 3
  uv run evalctl run --dataset "$ds" --template "$tpl" \
    --model "$MODEL" --provider "$PROVIDER" \
    --output "$OUT/$name" --concurrency 2 --temperature 0.0 --seed 42 "$@"
}

run squad-ci "$PACKS/squad-ci" fixtures/templates/qa_short.jinja --max-tokens 64
run squad-ci-b "$PACKS/squad-ci" fixtures/templates/qa_short.jinja --max-tokens 64 --temperature 0.2 --seed 7
run ag-news-ci "$PACKS/ag-news-ci" fixtures/templates/classification.jinja --max-tokens 32
run finqa-smoke "$PACKS/finqa-smoke" fixtures/templates/qa_short.jinja --max-tokens 64
run pubmedqa-smoke "$PACKS/pubmedqa-smoke" fixtures/templates/classification.jinja --max-tokens 16

SQUAD_A="$(basename "$(ls "$OUT/squad-ci"/*.json | head -1)" .json)"
SQUAD_B="$(basename "$(ls "$OUT/squad-ci-b"/*.json | head -1)" .json)"
uv run evalctl runs compare "$SQUAD_A" "$SQUAD_B" --metric squad_f1 \
  --allow-compatible --output "$OUT/compare-squad.json"

echo "Wrote live artifacts under $OUT (provider=$PROVIDER model=$MODEL)."
