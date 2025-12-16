#!/usr/bin/env bash
set -euo pipefail

# Resolve the root directory of the project
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Default or Environment provided BOARD_DIM, default to 11
BOARD_DIM="${BOARD_DIM:-11}"

# Rebuild the Hex engine with the specific dimension
# We clean first to ensure the preprocessor definition takes effect
(cd "$ROOT_DIR/hex" && make clean && make BOARD_DIM="$BOARD_DIM")

# Run the Hex engine from the current working directory (preserving relative paths for args)
"$ROOT_DIR/hex/hex" "$@"