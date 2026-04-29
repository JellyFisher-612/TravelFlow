#!/usr/bin/env bash
set -u

PROJECT_DIR="${PROJECT_DIR:-/home/users/lhy/TravelAgent/agent_langchain}"
REQ_DOC="${REQ_DOC:-$PROJECT_DIR/测试需求文档.md}"
LOOP_DIR="${LOOP_DIR:-$PROJECT_DIR/.codex-loop}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_DIR/scripts/codex-test-loop-prompt.md}"
MAX_ITERS="${MAX_ITERS:-${1:-10}}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_MODEL="${CODEX_MODEL:-}"
CODEX_PROFILE="${CODEX_PROFILE:-}"
CODEX_EXTRA_ARGS="${CODEX_EXTRA_ARGS:-}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
STOP_ON_CODEX_FAILURE="${STOP_ON_CODEX_FAILURE:-0}"
STOP_ON_VALIDATION_FAILURE="${STOP_ON_VALIDATION_FAILURE:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/codex-test-loop.sh [max_iters]

Environment overrides:
  PROJECT_DIR=/path/to/repo
  MAX_ITERS=5
  CODEX_MODEL=gpt-5.4
  CODEX_PROFILE=name
  CODEX_EXTRA_ARGS='--json'
  RUN_VALIDATION=0
  STOP_ON_CODEX_FAILURE=1
  STOP_ON_VALIDATION_FAILURE=1
  DRY_RUN=1

The loop starts a fresh `codex exec` session each iteration. Durable context is kept in:
  .codex-loop/progress.md
  .codex-loop/learnings.md
  .codex-loop/current_prompt.md
  .codex-loop/logs/

Use DRY_RUN=1 to build prompts and validate script behavior without launching Codex.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

cd "$PROJECT_DIR" || exit 1

if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
  echo "ERROR: codex CLI not found: $CODEX_BIN"
  exit 1
fi

if [ ! -f "$REQ_DOC" ]; then
  echo "ERROR: 测试需求文档不存在: $REQ_DOC"
  exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
  echo "ERROR: Prompt 文件不存在: $PROMPT_FILE"
  exit 1
fi

mkdir -p "$LOOP_DIR/logs"
touch "$LOOP_DIR/progress.md"
touch "$LOOP_DIR/learnings.md"

echo "=== Codex Test Improvement Loop ==="
echo "Project: $PROJECT_DIR"
echo "Requirement doc: $REQ_DOC"
echo "Prompt file: $PROMPT_FILE"
echo "Max iterations: $MAX_ITERS"
echo

run_validation() {
  local validation_ok=0

  if [ "$RUN_VALIDATION" = "0" ]; then
    echo "Validation disabled by RUN_VALIDATION=0."
    return 0
  fi

  if [ -f "package.json" ]; then
    if grep -q '"test"' package.json; then
      npm test || validation_ok=1
    fi
    if grep -q '"lint"' package.json; then
      npm run lint || validation_ok=1
    fi
    if grep -q '"build"' package.json; then
      npm run build || validation_ok=1
    fi
  fi

  if find . -maxdepth 4 -type f -name '*.py' | grep -q .; then
    python3 -m compileall -q agent_langchain || validation_ok=1
  fi

  if [ -f "pyproject.toml" ] || [ -f "pytest.ini" ] || [ -d "tests" ]; then
    if command -v pytest >/dev/null 2>&1; then
      PYTHONPATH=agent_langchain pytest || validation_ok=1
    else
      echo "pytest not installed; skipping pytest validation."
    fi
  fi

  if command -v ruff >/dev/null 2>&1; then
    ruff check . || validation_ok=1
  fi

  return "$validation_ok"
}

for i in $(seq 1 "$MAX_ITERS"); do
  echo
  echo "===================================="
  echo "Iteration $i / $MAX_ITERS"
  echo "===================================="

  if git diff --quiet && git diff --cached --quiet; then
    echo "Git working tree is clean."
  else
    echo "WARNING: Git working tree has existing changes. Codex must inspect and preserve them."
  fi

  iter_stamp="$(date '+%Y%m%d-%H%M%S')"
  iter_prompt="$LOOP_DIR/current_prompt.md"
  iter_log="$LOOP_DIR/logs/iteration-$i-$iter_stamp.log"
  iter_last="$LOOP_DIR/logs/iteration-$i-$iter_stamp-last-message.md"

  cp "$PROMPT_FILE" "$iter_prompt"
  {
    echo
    echo "## Current Iteration"
    echo
    echo "This is iteration $i of $MAX_ITERS."
    echo
    echo "Project directory:"
    echo "$PROJECT_DIR"
    echo
    echo "Test requirement document:"
    echo "$REQ_DOC"
    echo
    echo "Progress file:"
    echo "$LOOP_DIR/progress.md"
    echo
    echo "Learnings file:"
    echo "$LOOP_DIR/learnings.md"
    echo
    echo "Important: this may be a fresh Codex session. Use the durable files above as memory."
  } >> "$iter_prompt"

  echo "Running Codex..."

  codex_args=(exec --full-auto -C "$PROJECT_DIR" --output-last-message "$iter_last")
  if [ -n "$CODEX_MODEL" ]; then
    codex_args+=(-m "$CODEX_MODEL")
  fi
  if [ -n "$CODEX_PROFILE" ]; then
    codex_args+=(-p "$CODEX_PROFILE")
  fi
  if [ -n "$CODEX_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    extra_args=( $CODEX_EXTRA_ARGS )
    codex_args+=("${extra_args[@]}")
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1: skipping Codex execution." | tee "$iter_log"
    codex_status=0
  else
    "$CODEX_BIN" "${codex_args[@]}" "$(cat "$iter_prompt")" 2>&1 | tee "$iter_log"
    codex_status=${PIPESTATUS[0]}
  fi

  echo
  echo "Codex exit code: $codex_status"

  echo
  echo "Running post-iteration validation..."
  run_validation
  validation_status=$?

  if [ "$validation_status" -eq 0 ]; then
    validation_text="passed"
  else
    validation_text="failed"
  fi

  {
    echo
    echo "## Loop runner iteration $i - $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "- Codex exit code: $codex_status"
    echo "- Validation: $validation_text"
    echo "- Log: $iter_log"
    echo "- Last message: $iter_last"
  } >> "$LOOP_DIR/progress.md"

  if grep -q "ALL_TEST_REQUIREMENTS_DONE" "$LOOP_DIR/progress.md"; then
    echo
    echo "All test requirements marked done."
    break
  fi

  if [ "$codex_status" -ne 0 ] && [ "$STOP_ON_CODEX_FAILURE" = "1" ]; then
    echo "Stopping because Codex failed and STOP_ON_CODEX_FAILURE=1."
    break
  fi

  if [ "$validation_status" -ne 0 ] && [ "$STOP_ON_VALIDATION_FAILURE" = "1" ]; then
    echo "Stopping because validation failed and STOP_ON_VALIDATION_FAILURE=1."
    break
  fi
done

echo
echo "Loop finished."
echo "See:"
echo "$LOOP_DIR/progress.md"
echo "$LOOP_DIR/learnings.md"
