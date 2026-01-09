#!/usr/bin/env bash
set -euo pipefail

API_KEY=${1:-""}
ITEM=${2:-"banana"}
OUT=${3:-"data/external/${ITEM// /_}_search.json"}

if [ -z "$API_KEY" ]; then
  echo "Usage: $0 USDA_API_KEY [ITEM] [OUT]"
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

python src/ingest/usda_client.py --api-key "$API_KEY" --item "$ITEM" --out "$OUT"
