#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/eval_common.sh" biroad 95_5 "$@"
