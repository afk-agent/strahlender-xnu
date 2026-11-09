#!/bin/bash
# Compiles and runs the tests of the MetalFX upscaling of the presented image (needs a GPU with MetalFX, e.g. Apple silicon):
# metalfx_scale_test (spatial scaler: quality and cost) and metalfx_temporal_conv_test (the temporal scaler's jitter and motion
# conventions that runtime/nvn/src/nvn/taa.inc and metalfx_temporal.h rely on) and metalfx_interp_test (frame generation, macOS 26).
# Usage: run_metalfx_test.sh [IMAGE [OUT_DIR]]   (default: a synthetic test pattern; OUT_DIR receives the bilinear and MetalFX images)
set -euo pipefail
cd "$(dirname "$0")/../../../.."
mkdir -p work/recomp/test
N=scripts/recomp/runtime/nvn/src/nvn
/usr/bin/clang -O2 -Wall -fobjc-arc -I "$N" scripts/recomp/runtime/test/metalfx_scale_test.m -framework Metal -framework MetalFX \
  -framework Foundation -framework ImageIO -framework CoreGraphics -o work/recomp/test/metalfx_scale_test
/usr/bin/clang -O2 -Wall -fobjc-arc scripts/recomp/runtime/test/metalfx_temporal_conv_test.m -framework Metal -framework MetalFX \
  -framework Foundation -o work/recomp/test/metalfx_temporal_conv_test
/usr/bin/clang -O2 -Wall -fobjc-arc scripts/recomp/runtime/test/metalfx_interp_test.m -framework Metal -framework MetalFX \
  -framework Foundation -o work/recomp/test/metalfx_interp_test
work/recomp/test/metalfx_scale_test "$@"
work/recomp/test/metalfx_temporal_conv_test
work/recomp/test/metalfx_interp_test
