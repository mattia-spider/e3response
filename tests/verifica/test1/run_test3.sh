#!/bin/bash
# Test 3: Test 1 for deriv2nd_muB with muB_scale 30 and 300, one run at a time (RAM)
cd "$(dirname "$0")/../../.."
for s in 30 300; do
  E3R_X64=0 JAX_PLATFORMS=cpu /home/mattia/miniforge3/envs/mlplay/bin/python -W ignore tests/verifica/overfit.py \
    --model deriv2nd_muB --muB-scale $s --steps 3000 --out tests/verifica/test1/overfit_deriv2nd_muB_scale$s.json \
    > tests/verifica/test1/overfit_deriv2nd_muB_scale$s.log 2>&1
done
