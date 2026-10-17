#!/usr/bin/env bash
# Self-contained eval of IBM's Granite Decision LoRA (granite-4.1-30b) on the JEV Decision Index
# v0.3 public set. Single purpose: download the adapter, serve base+adapter on stock vLLM, run the
# card's single-forward-pass logit readout over the frozen suite, score. One GPU (80 GB) is enough.
#
#   bash eval_granite_decision_lora.sh
#
# Prerequisites: a Python env on PATH with vllm>=0.30, transformers>=4.44, huggingface_hub, httpx,
# numpy, git (+ one 80 GB GPU), and an HF login with access to the gated suite dataset
# (`huggingface-cli login` or HF_TOKEN). Everything else -- the decision-index kit, the frozen
# suite, the base model, the adapter -- is fetched automatically. Point PY at a specific interpreter
# if `python`/`vllm` are not the ones you want (e.g. PY=/path/to/venv/bin).
#
# Everything is relative to this script's own directory (we cd into it). Override via env:
#   BASE    base model: an HF repo id (auto-downloaded) or a local dir (default
#           ibm-granite/granite-4.1-30b)
#   SUITE   frozen Decision Index suite dir      (default ./suite-0.2)
#   KIT     the decision-index kit repo          (default ./decision-index)
#   WORK    where the run + results land         (default ./work)
#   ADAPTER a local adapter dir (else downloaded to ./adapter)
#   PORT, CONC, MAXLEN, EDITION (default 0.3.2 public panel)
set -uo pipefail

cd "$(dirname "$0")"                            # root everything at this script's dir
PY="${PY:-}"                                   # optional: dir of a specific python/vllm, else PATH
PYBIN="${PY:+$PY/}python"; VLLM="${PY:+$PY/}vllm"
WORK="${WORK:-./work}"
SUITE="${SUITE:-./suite-0.2}"
KIT="${KIT:-./decision-index}"
BASE="${BASE:-ibm-granite/granite-4.1-30b}"    # HF repo id (auto-downloaded) or a local dir
PORT="${PORT:-8231}"; CONC="${CONC:-48}"; MAXLEN="${MAXLEN:-32768}"
EDITION="${EDITION:-0.3.2}"                                           # v0.3.2 public panel == v0.3 public
NAME="granite41-30b-decision-lora"
RUN="$WORK/runs/$NAME"; mkdir -p "$RUN"
export HF_HUB_DISABLE_XET=1

# 0. kit + suite (auto-provisioned; nothing to install or build by hand).
#    kit: clone the public decision-index repo; it is used as an importable module (PYTHONPATH), no
#    pip step. suite: the kit downloads + verifies the frozen suite for the edition from HF.
if [ ! -d "$KIT" ]; then
  echo "[eval] cloning decision-index kit -> $KIT"
  git clone --depth 1 https://github.com/apolinario/decision-index "$KIT" || exit 1
fi
if [ ! -e "$SUITE/manifest.json" ]; then
  echo "[eval] downloading frozen suite (edition $EDITION) -> $SUITE"
  # The suite dataset is gated on HF: `huggingface-cli login` (or export HF_TOKEN) with access
  # granted to multimodalart/decision-index-suite-0.2, or point SUITE at a local copy.
  PYTHONPATH="$KIT${PYTHONPATH:+:$PYTHONPATH}" $PYBIN -m decision_index suite download \
    --edition "$EDITION" --dir "$SUITE" || {
      echo "[eval] suite download failed -- the suite dataset is gated; log in to HF"
      echo "       (huggingface-cli login / HF_TOKEN) with access to it, or set SUITE=<local copy>."
      exit 1; }
fi

# 1. adapter (standard PEFT LoRA, never merged). Skip if ADAPTER already points at a local dir.
ADAPTER="${ADAPTER:-}"
if [ -z "$ADAPTER" ]; then
  DL="./adapter"
  echo "[eval] downloading decision/granite-4.1-30b/lora -> $DL"
  $PYBIN - "$DL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("ibm-granite/granitelib-core-r1.0",
                  allow_patterns=["decision/granite-4.1-30b/lora/*"], local_dir=sys.argv[1])
PY
  ADAPTER="$DL/decision/granite-4.1-30b/lora"
fi
echo "[eval] adapter=$ADAPTER base=$BASE"

# 2. serve base + adapter on stock vLLM (NOT merged, NOT the aLoRA fork). vLLM orphans its
#    EngineCore, so setsid + reap the whole process group on exit.
setsid env HF_HUB_DISABLE_XET=1 $VLLM serve "$BASE" --served-model-name granite-base \
  --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules "decision-lora=$ADAPTER" \
  --tensor-parallel-size 1 --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 \
  --port "$PORT" --trust-remote-code > "$RUN/backend.log" 2>&1 &
echo $! > "$RUN/backend.ppid"; sleep 2
PGID=$(ps -o pgid= -p "$(cat "$RUN/backend.ppid")" 2>/dev/null | tr -d ' ')
trap 'kill -- -${PGID:-0} 2>/dev/null' EXIT   # frees the GPU when the run finishes or aborts
echo "[eval] serving on :$PORT (backend PGID=$PGID); waiting for the adapter ..."
for _ in $(seq 1 240); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "decision-lora" && break; sleep 5
done
curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "decision-lora" || { echo "[eval] adapter not served"; tail -40 "$RUN/backend.log"; exit 2; }

# 3. run the eval: card-exact prompt + letter labels, one forward pass per question.
echo "[eval] running decider (granitedec layout) -> $RUN/results.jsonl"
$PYBIN ./decider_granite_decision_lora.py \
  --base-url "http://127.0.0.1:$PORT" --model decision-lora --tokenizer "$BASE" \
  --suite "$SUITE" --out "$RUN/results.jsonl" --conc "$CONC" --maxlen "$MAXLEN" || exit 3

# 4. score at the requested public edition. The decision_index kit must be importable (pip install
# -e ./decision-index, or set PYTHONPATH); invoked as a module so no directory change is needed.
echo "[eval] scoring (edition $EDITION) -> $RUN"
PYTHONPATH="$KIT${PYTHONPATH:+:$PYTHONPATH}" $PYBIN -m decision_index score \
  --suite-dir "$SUITE" --edition "$EDITION" --results "$RUN/results.jsonl" --out "$RUN" || exit 4
$PYBIN - "$RUN/index.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[eval] Decision Index ({d.get('edition')} public) = {d.get('index')}  (raw {d.get('raw_index')})")
for a in d.get("areas", []): print(f"    {a['id']:10} skill={a['skill']:.4f}")
PY
echo "[eval] DONE -> $RUN"
