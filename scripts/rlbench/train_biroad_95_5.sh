#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/train_common.sh" biroad 95_5 "$@"
