#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/train_common.sh" 3dfa 50_50 "$@"
