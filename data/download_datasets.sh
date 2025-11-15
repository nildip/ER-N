#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(dirname "$0")"
ML_DIR="$DATA_DIR/movielens-1m"
ZIP_URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_PATH="$DATA_DIR/ml-1m.zip"

mkdir -p "$ML_DIR"

if [ -f "$ML_DIR/ratings.dat" ]; then
  echo "MovieLens 1M already downloaded at $ML_DIR/ratings.dat"
  exit 0
fi

echo "Downloading MovieLens 1M..."
if command -v curl >/dev/null 2>&1; then
  curl -L -o "$ZIP_PATH" "$ZIP_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ZIP_PATH" "$ZIP_URL"
else
  echo "Please install curl or wget to download datasets."
  exit 1
fi

echo "Extracting..."
unzip -o "$ZIP_PATH" -d "$DATA_DIR"
mv -f "$DATA_DIR/ml-1m/ratings.dat" "$ML_DIR/ratings.dat"
echo "Done. ratings file is at $ML_DIR/ratings.dat"
