#!/bin/bash
set -e

DEST_DIR="/shared/ssd/home/b-s-adhikari/nn-gpt/VJEPA2/data/ssv2"
mkdir -p "$DEST_DIR"

echo "Downloading Something-Something V2 from HuggingFace..."
huggingface-cli download olarian/something-something-v2 \
    --repo-type dataset \
    --local-dir "$DEST_DIR" \
    --local-dir-use-symlinks False \
    --include "*.json" "videos/*"

echo "Download complete! Now extracting..."
cd "$DEST_DIR/videos"
cat 20bn-something-something-v2-?? | tar zx -C "$DEST_DIR"

echo "Extraction complete! Videos are in $DEST_DIR/20bn-something-something-v2"
