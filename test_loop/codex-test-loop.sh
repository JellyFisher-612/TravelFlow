#!/usr/bin/env bash
set -u

PROJECT_DIR="${PROJECT_DIR:-/home/users/lhy/TravelAgent/agent_langchain}"
exec "$PROJECT_DIR/scripts/codex-test-loop.sh" "$@"
