#!/bin/bash
cd "$(dirname "$0")"
EXTRA=1 EXTRA_COMPS=jointavg,chainm python3 -W ignore stack_pipeline.py 2 v14 0.5 cl succ_scores_cv_h.pkl test_succ_h.pkl > ../work/stack_v14.log 2>&1
EXTRA_COMPS=jointavg,chainm SUCC_TR=succ_scores_cv_h.pkl SUCC_TE=test_succ_h.pkl P1_SIM=sim_P1_v14.npy P1_TE=test_P1_v14.npy python3 -W ignore hgb_stage.py hgb3hm v14_oof_stage2.npy v14_pte_stage2.npy v14_dec_tr_stage2.npy v14_dec_te_stage2.npy > ../work/hgb3h.log 2>&1
echo CHAIN_DONE >> ../work/hgb3h.log
