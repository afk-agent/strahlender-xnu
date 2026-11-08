#!/bin/bash
# Compiles and runs the test of the MetalFX upscaling of the presented image (needs a GPU with MetalFX, e.g. Apple silicon).
# Usage: run_metalfx_test.sh [IMAGE [OUT_DIR]]   (default: a synthetic test pattern; OUT_DIR receives the bilinear and MetalFX images)
set -euo pipefail
cd "$(dirname "$0")/../../../.."
mkdir -p work/recomp/test
N=scripts/recomp/runtime/nvn/src/nvn
/usr/bin/clang -O2 -Wall -fobjc-arc -I "$N" scripts/recomp/runtime/test/metalfx_scale_test.m -framework Metal -framework MetalFX \
  -framework Foundation -framework ImageIO -framework CoreGraphics -o work/recomp/test/metalfx_scale_test
work/recomp/test/metalfx_scale_test "$@"
