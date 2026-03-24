#!/usr/bin/env bash
# Run README pipeline Steps 5–8: Eval 1 (H2, optional H1), Eval 2 (LLM QA), Eval 3 (MKQA).
# Starts/stops llama-server on port 8080 as needed (make server-global / make server-fire).
#
# Usage (from anywhere):
#   ./run_eval_pipeline.sh
#   ./run_eval_pipeline.sh --limit 10          # smoke test
#   ./run_eval_pipeline.sh --skip-step6          # skip H1 Fire vs Global on Hindi
#   ./run_eval_pipeline.sh --from-step 6         # resume after Step 5 finished (skips H2 + copy)
#   ./run_eval_pipeline.sh --pl-merge            # Polish only: eval1-h2 + eval2-llm with --merge (rest skipped)
#
# Expect long wall time for a full run (hours). Consider tmux or nohup.

set -euo pipefail

DOCUNATIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOCUNATIVE_ROOT"

LIMIT="${LIMIT:-}"
SKIP_STEP6="${SKIP_STEP6:-0}"
FROM_STEP="${FROM_STEP:-5}"
PL_MERGE="${PL_MERGE:-0}"

usage() {
  cat <<'EOF'
Run README Steps 5–8 (Eval 1 H2, optional H1, Eval 2 LLM QA, Eval 3 MKQA).

Usage: ./run_eval_pipeline.sh [options]

  --limit N          Pass --limit to eval.evaluate and eval_mkqa (smoke tests)
  --skip-step6       Skip Eval 1 H1 (Fire vs Global on Hindi)
  --from-step N      Start at step N (5–8). Use 6 after Step 5 already completed.
  --pl-merge         Only run Polish incremental eval (eval1-h2 + eval2-llm with --language pl --merge).
                     Skips the full Steps 5–8 pipeline. Requires make server-global / llama-server.
  -h, --help         Show this help

Environment:
  SKIP_STEP6=1       Same as --skip-step6
  LIMIT=N            Same as --limit (CLI wins if both set)
  FROM_STEP=N        Same as --from-step (CLI wins)
  PL_MERGE=1         Same as --pl-merge

Run from docunative/ or any path; the script cds to its directory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="${2:?--limit requires a number}"
      shift 2
      ;;
    --skip-step6)
      SKIP_STEP6="1"
      shift
      ;;
    --from-step)
      FROM_STEP="${2:?--from-step requires 5, 6, 7, or 8}"
      shift 2
      ;;
    --pl-merge)
      PL_MERGE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${PL_MERGE}" != "1" ]]; then
  case "${FROM_STEP}" in
    5|6|7|8) ;;
    *)
      echo "ERROR: --from-step must be 5, 6, 7, or 8 (got ${FROM_STEP})" >&2
      exit 1
      ;;
  esac
fi

LOG_DIR="${DOCUNATIVE_ROOT}/eval/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/llama_server.log"

stop_server() {
  local pids
  pids="$(lsof -ti :8080 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Stopping process(es) on port 8080: ${pids}"
    kill -TERM ${pids} 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti :8080 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
      echo "Force killing remaining: ${pids}"
      kill -KILL ${pids} 2>/dev/null || true
      sleep 1
    fi
  fi
}

wait_for_server() {
  local max_attempts="${1:-180}"
  local i=0
  echo "Waiting for llama-server on http://localhost:8080/health (timeout ${max_attempts}s) ..."
  while (( i < max_attempts )); do
    if uv run python -c \
      "from pipeline.generate import check_server_health; raise SystemExit(0 if check_server_health() else 1)" \
      2>/dev/null; then
      echo "llama-server is healthy."
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "ERROR: llama-server did not become healthy within ${max_attempts}s (see ${SERVER_LOG})" >&2
  return 1
}

start_global() {
  echo "=== Starting Global model (make server-global) ==="
  stop_server
  : > "${SERVER_LOG}"
  make server-global >> "${SERVER_LOG}" 2>&1 &
  SERVER_STARTED=1
  # Large GGUF can take several minutes to mmap / warm up on first load
  wait_for_server 600
}

start_fire() {
  echo "=== Starting Fire model (make server-fire) ==="
  stop_server
  : > "${SERVER_LOG}"
  make server-fire >> "${SERVER_LOG}" 2>&1 &
  SERVER_STARTED=1
  wait_for_server 600
}

cleanup() {
  if [[ "${SERVER_STARTED:-0}" -eq 1 ]]; then
    echo ""
    echo "Stopping llama-server (pipeline finished or interrupted)..."
    stop_server
  fi
}

trap cleanup EXIT INT TERM

SERVER_STARTED=0

# Build command in a single array so optional --limit works with `set -u` (empty
# "${EVAL_ARGS[@]}" is treated as unbound on some Bash versions).
run_evaluate() {
  local cmd=(uv run python -m eval.evaluate "$@")
  if [[ -n "${LIMIT:-}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  "${cmd[@]}"
}

run_mkqa() {
  local lim=100
  if [[ -n "${LIMIT:-}" ]]; then
    lim="${LIMIT}"
  fi
  uv run python -m eval.eval_mkqa --model Global --limit "${lim}"
}

echo "DocuNative — eval pipeline Steps 5–8"
echo "Working directory: ${DOCUNATIVE_ROOT}"
if [[ -n "${LIMIT}" ]]; then
  echo "Limit: ${LIMIT} (smoke test mode)"
fi
if [[ "${FROM_STEP}" != "5" ]]; then
  echo "Resuming from Step ${FROM_STEP} (steps before that are skipped)"
fi
if [[ "${SKIP_STEP6}" == "1" ]]; then
  echo "Skipping Step 6 (Eval 1 H1 — Fire vs Global on Hindi)"
fi
if [[ "${PL_MERGE}" == "1" ]]; then
  echo "Mode: --pl-merge (Polish incremental eval only; full pipeline skipped)"
fi
echo ""

# --- Polish-only merge (Eval 1 H2 + Eval 2 LLM) ---
if [[ "${PL_MERGE}" == "1" ]]; then
  start_global
  echo ""
  echo "=== Polish merge — Eval 1 H2 (template QA) ==="
  run_evaluate \
    --qa dataset/output/qa_pairs.jsonl \
    --docs dataset/output \
    --model Global \
    --run-name eval1-h2 \
    --language pl \
    --merge
  echo ""
  echo "=== Polish merge — Eval 2 LLM QA ==="
  run_evaluate \
    --qa dataset/output/qa_pairs_llm.jsonl \
    --docs dataset/output \
    --model Global \
    --run-name eval2-llm \
    --language pl \
    --merge
  echo ""
  echo "Refreshing eval/results/eval_results.jsonl for MKQA comparison..."
  cp -f eval/results/eval_results_eval1-h2.jsonl eval/results/eval_results.jsonl
  echo ""
  echo "Done (--pl-merge). eval/results/eval_results_eval{1-h2,2-llm}.jsonl updated with Polish rows."
  SERVER_STARTED=0
  stop_server
  exit 0
fi

# --- Step 5 — Eval 1 H2 ---
if [[ "${FROM_STEP}" == "5" ]]; then
  start_global
  echo ""
  echo "=== Step 5 — Eval 1 H2 (Global, all languages) ==="
  run_evaluate \
    --qa dataset/output/qa_pairs.jsonl \
    --docs dataset/output \
    --model Global \
    --run-name eval1-h2

  echo ""
  echo "Preparing synthetic results for Eval 3 MKQA comparison report..."
  cp -f eval/results/eval_results_eval1-h2.jsonl eval/results/eval_results.jsonl
else
  echo "=== Steps 1–4 / Step 5 — skipped (resume) ==="
  if [[ ! -f eval/results/eval_results_eval1-h2.jsonl ]]; then
    echo "WARNING: eval/results/eval_results_eval1-h2.jsonl not found — MKQA comparison may be wrong." >&2
  fi
  if [[ -f eval/results/eval_results_eval1-h2.jsonl ]]; then
    echo "Refreshing eval/results/eval_results.jsonl for MKQA comparison from eval1-h2 run..."
    cp -f eval/results/eval_results_eval1-h2.jsonl eval/results/eval_results.jsonl
  fi
fi

# --- Step 6 — Eval 1 H1 (optional) ---
if [[ "${SKIP_STEP6}" != "1" ]] && [[ "${FROM_STEP}" -le 6 ]]; then
  echo ""
  echo "=== Step 6 — Eval 1 H1 (Fire vs Global, Hindi only) ==="
  start_fire
  run_evaluate \
    --qa dataset/output/qa_pairs.jsonl \
    --docs dataset/output \
    --model Fire --language hi \
    --run-name eval1-h1-fire

  start_global
  run_evaluate \
    --qa dataset/output/qa_pairs.jsonl \
    --docs dataset/output \
    --model Global --language hi \
    --run-name eval1-h1-global
elif [[ "${SKIP_STEP6}" == "1" ]] && [[ "${FROM_STEP}" -le 6 ]]; then
  echo ""
  echo "=== Step 6 — skipped (per --skip-step6; Global still up after Step 5) ==="
elif [[ "${FROM_STEP}" -gt 6 ]]; then
  echo ""
  echo "=== Step 6 — skipped (resume from step ${FROM_STEP}) ==="
fi

# Resume from 7/8: no guarantee anything is listening on 8080
if [[ "${FROM_STEP}" -ge 7 ]]; then
  start_global
fi

# --- Step 7 — Eval 2 LLM QA ---
if [[ "${FROM_STEP}" -le 7 ]]; then
  echo ""
  echo "=== Step 7 — Eval 2 LLM QA ==="
  run_evaluate \
    --qa dataset/output/qa_pairs_llm.jsonl \
    --docs dataset/output \
    --model Global \
    --run-name eval2-llm
fi

# --- Step 8 — MKQA ---
echo ""
echo "=== Step 8 — Eval 3 MKQA (zh, pl) ==="
run_mkqa

echo ""
echo "Done. Outputs under eval/results/ (see README for filenames)."
SERVER_STARTED=0
stop_server
