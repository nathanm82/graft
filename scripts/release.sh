#!/usr/bin/env bash
# Tag a release after bumping graft/__about__.py.
# Usage: scripts/release.sh 0.3.0
set -euo pipefail

version="${1:?usage: scripts/release.sh X.Y.Z}"

if ! grep -q "\"${version}\"" graft/__about__.py; then
    echo "graft/__about__.py does not declare ${version}; bump it first." >&2
    exit 1
fi

git tag -a "v${version}" -m "graft ${version}"
echo "Tagged v${version}. Push it with: git push origin v${version}"
