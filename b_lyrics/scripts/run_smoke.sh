#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
env_dir="${CARE_ENV_DIR:-$PWD}"
"$env_dir/.venv-kiki/bin/python" -m group_b.features kiki query --start-index 41 --limit 1
"$env_dir/.venv-b/bin/python" -m group_b.features text query --start-index 41 --limit 1
"$env_dir/.venv-b/bin/python" -m group_b.features vocals query --start-index 41 --limit 1
"$env_dir/.venv-kiki/bin/python" -m group_b.features kiki reference --start-index 41 --limit 1
"$env_dir/.venv-b/bin/python" -m group_b.features text reference --start-index 41 --limit 1
"$env_dir/.venv-kiki/bin/python" -m group_b.features kiki jamendo --limit 1
"$env_dir/.venv-b/bin/python" -m group_b.features asr jamendo --limit 1
