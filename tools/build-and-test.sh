#!/bin/sh
# Rebuild the extension and prove the build actually picked up the sources, then run the suite.
#
# Why this exists: in this workspace the CMake tree builds gufo-src *and* the extension from
# copies, `rsync -a` preserves mtimes, and ninja then considers the extension objects current.
# Measurements were taken against stale binaries three times before this script existed. The
# script therefore deletes the extension objects and checks a marker literal is present in the
# produced binary before it lets the tests run.
#
# Usage: tools/build-and-test.sh
#   REPO_SOURCE=<dir>  rsync this into REPO first (e.g. the git checkout)
#   REPO=/work/ext-build           the copy the build compiles
#   BUILD=/work/ext-gpu-test       the configured CMake build tree
#   MARKER="engine: registered"    a literal that must appear in the binary
set -eu

REPO=${REPO:-/work/ext-build}
BUILD=${BUILD:-/work/ext-gpu-test}
TARGET=${TARGET:-ling3_air_forward_test}
BIN="$BUILD/extensions/build/ling3_air/build/$TARGET"
MARKER=${MARKER:-"engine: chunked prefill failed"}

if [ -n "${REPO_SOURCE:-}" ]; then
  echo "build-and-test: syncing $REPO_SOURCE -> $REPO"
  rsync -a --exclude .git "$REPO_SOURCE"/ "$REPO"/
fi

echo "build-and-test: removing extension objects so ninja must recompile them"
find "$BUILD/extensions" -name '*.o' -path '*ling3*' -delete 2>/dev/null || true

echo "build-and-test: building $TARGET"
cmake --build "$BUILD" --target "$TARGET" >/dev/null

if [ ! -x "$BIN" ]; then
  echo "build-and-test: $BIN missing after the build" >&2
  exit 1
fi
if ! grep -qF -- "$MARKER" "$BIN"; then
  echo "build-and-test: marker '$MARKER' is absent from $BIN" >&2
  echo "build-and-test: the build did not pick up the sources; do not trust any measurement" >&2
  exit 1
fi
echo "build-and-test: marker present, build is current"

ctest --test-dir "$BUILD" -R '^ling3_air'
