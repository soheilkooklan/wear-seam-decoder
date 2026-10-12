#!/bin/bash
cd "$(dirname "$0")"
export STACK_LW=3
EXTRA=1 EXTRA_COMPS=jointavg,chainm python3 -W ignore stack_pipeline.py 2 v15f 0.5 cl succ_scores_cv_hli.pkl test_succ_hli.pkl > ../work/stack_v15f.log 2>&1
EXTRA_COMPS=jointavg,chainm SUCC_TR=succ_scores_cv_hli.pkl SUCC_TE=test_succ_hli.pkl P1_SIM=sim_P1_v15f.npy P1_TE=test_P1_v15f.npy python3 -W ignore hgb_stage.py hgb3i3f v15f_oof_stage2.npy v15f_pte_stage2.npy v15f_dec_tr_stage2.npy v15f_dec_te_stage2.npy > ../work/hgb3i3f.log 2>&1
echo CHAIN_DONE >> ../work/hgb3i3f.log
