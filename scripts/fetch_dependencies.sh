#!/usr/bin/env bash
# Fetch the vendored dependency: Alpaka 1.3.0.
#
# 1.3.0 is required. 1.0.0 fails to compile its CUDA buffer header under
# nvcc 12.4 with "Idx is not a template" — the use of alpaka::Idx<T> there is
# unqualified and collides with the surrounding lookup. 1.3.0 qualifies it.
set -euo pipefail

VERSION="1.3.0"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/third_party/alpaka-$VERSION"

if [ -d "$DEST" ]; then
    echo "already present: $DEST"
    exit 0
fi

mkdir -p "$HERE/third_party"
git clone --depth 1 --branch "$VERSION" \
    https://github.com/alpaka-group/alpaka.git "$DEST"

echo "fetched Alpaka $VERSION into $DEST"
