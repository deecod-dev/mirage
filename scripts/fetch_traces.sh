#!/usr/bin/env bash
# Download and unpack the raw agent traces into artifacts/.
#   bash scripts/fetch_traces.sh                 # every experiment
#   bash scripts/fetch_traces.sh E4_fingerprint_v2 E4_2wiki_v3
set -euo pipefail
cd "$(dirname "$0")/.."
base="${MIRAGE_TRACES_URL:-https://github.com/deecod-dev/mirage/releases/download/v1.0}"
if (( $# )); then names=("$@"); else names=($(awk '{sub(/\.tar\.gz$/, "", $2); print $2}' data/SHA256SUMS)); fi
mkdir -p artifacts .downloads
for name in "${names[@]}"; do
  file="$name.tar.gz"
  [[ -f ".downloads/$file" ]] || curl -fL --retry 3 -o ".downloads/$file" "$base/$file"
  (cd .downloads && grep " $file\$" ../data/SHA256SUMS | sha256sum -c -)
  tar -xzf ".downloads/$file" -C artifacts
done
echo "traces unpacked under artifacts/"
