#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
env_dir="${CARE_ENV_DIR:-$PWD}"
case "${1:-}" in
  covers80)
    for role in query reference; do
      "$env_dir/.venv-kiki/bin/python" -m group_b.features kiki "$role"
      "$env_dir/.venv-b/bin/python" -m group_b.features text "$role"
    done
    "$env_dir/.venv-b/bin/python" -m group_b.features vocals query
    ;;
  jamendolyrics)
    "$env_dir/.venv-kiki/bin/python" -m group_b.features kiki jamendo
    "$env_dir/.venv-b/bin/python" -m group_b.features asr jamendo
    ;;
  *) echo "Usage: bash run_features.sh {covers80|jamendolyrics}" >&2; exit 2 ;;
esac
"$env_dir/.venv-b/bin/python" -m group_b.handoff index
