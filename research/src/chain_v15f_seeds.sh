#!/bin/bash
# Seed replicates of the pseudo-labelled chain (stage-2 MLP seeds only; stage 3 is deterministic).
cd "$(dirname "$0")"
export STACK_LW=3
cp ../work/tiles_fixed.pkl ../work/tiles.pkl
for s in 1 2; do
  cp ../work/sim_P1_v15f.npy ../work/sim_P1_v15fs$s.npy; cp ../work/test_P1_v15f.npy ../work/test_P1_v15fs$s.npy
  SEED_OFF=$((s * 1000)) EXTRA=1 EXTRA_COMPS=jointavg,chainm python3 -W ignore stack_pipeline.py 2 v15fs$s 0.5 cl succ_scores_cv_hli.pkl test_succ_hli.pkl > ../work/stack_v15fs$s.log 2>&1
  EXTRA_COMPS=jointavg,chainm SUCC_TR=succ_scores_cv_hli.pkl SUCC_TE=test_succ_hli.pkl P1_SIM=sim_P1_v15fs$s.npy P1_TE=test_P1_v15fs$s.npy python3 -W ignore hgb_stage.py hgb3i3fs$s v15fs${s}_oof_stage2.npy v15fs${s}_pte_stage2.npy v15fs${s}_dec_tr_stage2.npy v15fs${s}_dec_te_stage2.npy > ../work/hgb3i3fs$s.log 2>&1
done
cp ../work/tiles_orig.pkl ../work/tiles.pkl
echo CHAIN_DONE >> ../work/hgb3i3fs2.log
