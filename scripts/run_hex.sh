#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../hex"
make
./hex | head -n 60
