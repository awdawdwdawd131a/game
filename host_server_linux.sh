#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
./run_linux_mac.sh --server --port "${1:-27015}"
