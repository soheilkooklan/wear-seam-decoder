# Order of the steps

The steps below are the ones used for the final system. Scripts in `kaggle/`
ran as Kaggle notebooks with the competition data attached; their output
archives were copied to the project folder. All other commands run from
`research/src`. File tags: chain A = `v11` / `hgb3h`, chain B = `v14` /
`hgb3hm`, chain C = `v15f` / `hgb3i3f`; link models f (first pass), h
(rescorer), h3 (Hungarian), hl9 (video bi-encoder, 9 seeds), hli (final link
model), hliq (seam features).

1. Data. `kaggle/export_pack.py` writes the compressed video (224 principal
   components, 8-bit) and the test arrays. `python3 prepare.py` unpacks them and
   writes `work/tiles.pkl` and `work/imu_feats_exact.npy`. It needs `ref/cache/`
   (see the README).
2. Window models.
   - `python3 run_video.py snorm` (video MLP, out-of-fold)
   - `python3 bg_models.py` (IMU MLP and IMU boosting, out-of-fold, and the test predictions of the IMU boosting model)
   - `python3 train_final_v1.py` (test predictions of the video and IMU MLPs)
   - `kaggle/deep_models.py`, then `python3 integrate_deep.py` (video transformer and IMU CNN)
   - `kaggle/joint_model.py` and `kaggle/joint2_model.py`, then `python3 integrate_joint.py <archive>`
   - interactive: mean of the two joint networks (`oof_jointavg.npy`, `test_jointavg.npy`) and the fused stage-1 probabilities (`sim_P1_<tag>.npy`, `test_P1_<tag>.npy`, weights 0.2, 0.2, 0.2, 0.3, 0.1)
3. Link models.
   - `kaggle/contrast_full.py` (contrastive embeddings, into `work/contrast/`)
   - `python3 run_succ6_cv.py f 2` and `python3 run_succ6_final.py f` (first pass)
   - `python3 rescore.py h` (rescorer)
   - `kaggle/link2_model.py`, `kaggle/link3_model.py`, `kaggle/link4_model.py`, then `python3 merge_link.py <three archives>` (video bi-encoder, into `work/link9/`)
   - `kaggle/imu_link_model.py`, then `python3 integrate_imu_link.py <archive>` (IMU bi-encoder)
   - `HUNG=1 python3 rescore.py h3 succ_scores_cv_h.pkl test_succ_h.pkl`
   - `LINK=1 LINK_DIR=link9 python3 rescore.py hl9 succ_scores_cv_h.pkl test_succ_h.pkl`
   - `LINK=1 LINK_DIR=link9 ILINK=1 python3 rescore.py hli succ_scores_cv_h.pkl test_succ_h.pkl`
   - the same with `IMUQ=1` and tag `hliq` for the seam features
4. Sequence model. `kaggle/chain_model.py`, then `python3 integrate_chainmodel.py <archive> chainm`.
5. Stacking. `bash chain_h.sh` (A), `bash chain_hm.sh` (B) and `bash chain_v15f.sh`
   (C, with `work/tiles.pkl` replaced by the relabelled tile table while it
   runs). Variants: `chain_v15.sh` (original labels), `chain_v15x.sh` (spans
   left out, `TILES=tiles_excl.pkl`), `chain_v15q.sh` (seam links),
   `chain_v15f_seeds.sh` (seed replicates).
6. Test predictions.
   `BETA=1.5 IMU_W=1.0 LO=75 HI=95 python3 make_mf_sub.py v15 v22 test_succ_hli.pkl 0.6 hgb3h_pte,hgb3hm_pte,hgb3i3f_pte`
   (final submission). The later check with the spans left out and bounds
   85/100 (Section 4.3):
   `BETA=1.5 IMU_W=1.0 LO=85 HI=100 python3 make_mf_sub.py v15 v24 test_succ_hli.pkl 0.6 hgb3h_pte,hgb3hm_pte,hgb3i3x_pte`
7. Numbers of the report. `python3 ../analysis/paper_numbers.py` (most tables),
   `seam_eval.py` and `seam_spread.py` (Table 3), `nested_check.py`,
   `seed_eval.py`, `protocol_checks.py` and `protocol_checks2.py` (Section 4.7
   and Table 7), `v24_check.py` (simulation score of submission v24).

The analysis scripts read the files written by the earlier steps. If one is
missing, they stop with an error that names the file and the project folder.
`decode_lib.py` holds the earlier cluster decoder that the stacking stages use
for one of their features (Section 3.5); the decoder of the final system is the
one in `../decode.py`. The two are separate on purpose: `decode_lib.py` is kept
as it was run, so that the stacking features can be rebuilt unchanged.
