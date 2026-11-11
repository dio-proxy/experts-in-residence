#!/bin/sh
# Trains every configuration in PREREGISTRATION.md, most important first. Skips finished runs.
run() { name=$1; shift; [ -f runs/$name/train.json ] || python train.py --name $name "$@" > runs_$name.log 2>&1 && rm -f runs_$name.log; }
run flat        --router flat
run tree        --router tree
run flat-l0.1   --router flat --lam 0.1
run tree-l0.1   --router tree --lam 0.1
run flat-l0.3   --router flat --lam 0.3
run tree-l0.3   --router tree --lam 0.3
run flat-l0.03  --router flat --lam 0.03
run tree-l0.03  --router tree --lam 0.03
run tree-slow   --router tree --lam 0.1 --slow 0.9
run flat-seed1  --router flat --seed 1
run dense       --router dense
