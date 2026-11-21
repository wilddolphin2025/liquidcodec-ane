#!/usr/bin/env bash
# Compile .mlpackage files to .mlmodelc. macOS only (xcrun coremlcompiler).
# Core ML also loads .mlpackage directly on macOS 12 and later; compiling is optional.
set -euo pipefail
ROOT="${1:-models/stream}"
if ! command -v xcrun >/dev/null; then
  echo "xcrun is not on this machine. Load the .mlpackage files directly, or run this script on a Mac." >&2
  exit 1
fi
find "$ROOT" -name '*.mlpackage' -type d | while read -r pkg; do
  out="${pkg%.mlpackage}.mlmodelc"
  echo "compile $pkg -> $out"
  rm -rf "$out"
  xcrun coremlcompiler compile "$pkg" "$(dirname "$out")"
done
echo "compiled packages under $ROOT"
