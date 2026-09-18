#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -m data_processing.peract2_to_zarr "$@" --ratio 50_50
