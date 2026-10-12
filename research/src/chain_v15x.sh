#!/bin/bash
cd "$(dirname "$0")"
export STACK_LW=3
export TILES=tiles_excl.pkl
EXTRA=1 EXTRA_COMPS=jointavg,chainm python3 -W ignore stack_pipeline_x.py 2 v15x 0.5 cl succ_scores_cv_hli.pkl test_succ_hli.pkl > ../work/stack_v15x.log 2>&1
EXTRA_COMPS=jointavg,chainm SUCC_TR=succ_scores_cv_hli.pkl SUCC_TE=test_succ_hli.pkl P1_SIM=sim_P1_v15x.npy P1_TE=test_P1_v15x.npy python3 -W ignore hgb_stage_x.py hgb3i3x v15x_oof_stage2.npy v15x_pte_stage2.npy v15x_dec_tr_stage2.npy v15x_dec_te_stage2.npy > ../work/hgb3i3x.log 2>&1
echo CHAIN_DONE >> ../work/hgb3i3x.log
