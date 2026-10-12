#!/bin/bash
cd "$(dirname "$0")"
export STACK_LW=3
EXTRA=1 EXTRA_COMPS=jointavg,chainm python3 -W ignore stack_pipeline.py 2 v15q 0.5 cl succ_scores_cv_hliq.pkl test_succ_hliq.pkl > ../work/stack_v15q.log 2>&1
EXTRA_COMPS=jointavg,chainm SUCC_TR=succ_scores_cv_hliq.pkl SUCC_TE=test_succ_hliq.pkl P1_SIM=sim_P1_v15q.npy P1_TE=test_P1_v15q.npy python3 -W ignore hgb_stage.py hgb3i3q v15q_oof_stage2.npy v15q_pte_stage2.npy v15q_dec_tr_stage2.npy v15q_dec_te_stage2.npy > ../work/hgb3i3q.log 2>&1
echo CHAIN_DONE >> ../work/hgb3i3q.log
cp ../work/tiles_orig.pkl ../work/tiles.pkl
