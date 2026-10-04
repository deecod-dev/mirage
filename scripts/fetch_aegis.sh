#!/usr/bin/env bash
# Clone the external AEGIS benchmark at the exact commit used in the paper.
set -euo pipefail
cd "$(dirname "$0")/.."
commit=18fecc5641235a67299cd832632839c9c80244d6
if [[ ! -d external/AEGIS/.git ]]; then
  git clone https://github.com/kfq20/AEGIS.git external/AEGIS
fi
git -C external/AEGIS checkout --quiet "$commit"
echo "external/AEGIS at $(git -C external/AEGIS rev-parse HEAD)"
