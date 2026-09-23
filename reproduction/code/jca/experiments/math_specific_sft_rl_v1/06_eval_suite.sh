#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" -B "$EXPERIMENT_ROOT/evaluate_main_table.py" run "$@"
