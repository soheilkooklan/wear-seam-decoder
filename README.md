# Code for the WEAR 2026 report

Code for the report "An Error Budget for Shuffled Multimodal Windows in the 3rd
WEAR Challenge: Stacking, Count Constraints and Seam Matching"
(Soheil Kooklan, 2026).

The repository has two parts.

## 1. Seam distance and decoder

| File | Content |
|---|---|
| `seam.py` | Seam distance between 1 s accelerometer windows of one limb and the rank of the true successor (Table 3 of the report) |
| `decode.py` | Participant graph (video kNN, same-limb IMU kNN, successor links) and the decoder with a soft count constraint (Sections 3.4 and 3.6) |

Requirements: Python 3.9 or later, NumPy, SciPy (`requirements.txt`; tested
with NumPy 2.4 and SciPy 1.17).

```
python seam.py                 # synthetic check of the seam distance
python reproduce_example.py    # decoder on two example participants
python tests/test_seam.py      # tests (they also run under pytest)
python tests/test_decode.py
```

`examples/` holds, for two training participants (4 and 21), the final unary
log probabilities, the participant graph and the labels of the final system in
simulation. `reproduce_example.py` decodes them and reports how many labels
differ from the final system (none). The files hold model outputs and a graph,
no competition data.

With the challenge data, `seam_distance` and `decode` are applied to one
participant at a time. `seam_distance` builds a dense n x n matrix, which is
small for one participant but should not be used for several participants at
once. Pass the limb ids so that cross-limb pairs are excluded.

Default arguments of `build_graph` and `decode` are the values of the final
submission: k = 10, IMU weight 1, link weight 10, sum symmetrisation,
beta = 1.5, l = 75, h = 95, 30 iterations, step 0.15, bias updates from the
sixth iteration. `lo` and `hi` also accept one bound per class. A bias cannot
exceed step x (iterations - 5) = 3.75 in absolute value; `max_bias` clips it
further if given (not used in the report).

## 2. Research scripts

`research/` holds the scripts that produced the results of the report, as they
were used. They document every trained part, but they are not a one-command
reproduction: paths are relative to a project folder (`WEAR_ROOT`, with
`pack/`, `work/` and `ref/` inside it), the competition data are needed, and a
few small steps were run as short interactive commands and are not included
(averaging the two joint networks, writing the fused stage-1 probabilities with
the weights of Appendix A, and writing the tile tables with relabelled or
excluded spans). The cache in `ref/cache/` (the list of recordings and one
accelerometer array per recording, samples x 4 limbs x 3 axes) is built from the
training files of the competition by a step that is not included.

`research/STEPS.md` gives the order of the steps and the main commands, and
`research/requirements.txt` the library versions. The trained models are not
released; every number of the report comes from models trained by these
scripts. Without the competition data, the part that can be checked is the
decoder of part 1 on the example participants.

File tags in the scripts name the experiments in the order they were made:
`v11`, `v14` and `v15f` are stage 2 of chains A, B and C, `hgb3h`, `hgb3hm` and
`hgb3i3f` their stage 3, and `f`, `h`, `hli` and `hliq` the link models
(first pass, rescorer, final link model, link model with seam features).
`STEPS.md` lists all tags. When a script needs a file that an earlier step has
not written, it stops with a message that names the file and the project
folder.

| Step | Scripts |
|---|---|
| Data pack, compressed video (Kaggle) | `kaggle/export_pack.py`, `src/prepare.py` |
| Video and IMU window models | `kaggle/deep_models.py`, `src/run_video.py`, `src/bg_models.py`, `src/train_final_v1.py`, `src/integrate_deep.py` |
| Joint networks (Kaggle) | `kaggle/joint_model.py`, `kaggle/joint2_model.py`, `src/integrate_joint.py` |
| First-pass link model | `kaggle/contrast_full.py`, `src/contrast_lib.py`, `src/successor.py`, `src/run_succ6_cv.py`, `src/run_succ6_final.py` |
| Bi-encoders (Kaggle) | `kaggle/link2_model.py`, `kaggle/link3_model.py`, `kaggle/link4_model.py`, `src/merge_link.py`, `kaggle/imu_link_model.py`, `src/integrate_imu_link.py` |
| Rescorer | `src/rescore.py` (environment flags `LINK=1`, `ILINK=1`, `HUNG=1` and `IMUQ=1` add the video bi-encoder, IMU bi-encoder, Hungarian and seam features) |
| Sequence model (Kaggle) | `kaggle/chain_model.py`, `src/integrate_chainmodel.py` |
| Stacking chains A, B, C | `src/stage2.py`, `src/stack_pipeline.py`, `src/hgb_stage.py`, `src/chain_h.sh`, `src/chain_hm.sh`, `src/chain_v15f.sh` |
| Label and seam variants, seed replicates | `src/chain_v15.sh`, `src/chain_v15x.sh`, `src/chain_v15q.sh`, `src/chain_v15f_seeds.sh`, `src/stack_pipeline_x.py`, `src/hgb_stage_x.py` |
| Test predictions | `src/make_mf_sub.py` |
| Numbers of the report | `analysis/paper_numbers.py`, `analysis/seam_eval.py`, `analysis/seam_spread.py`, `analysis/nested_check.py`, `analysis/seed_eval.py`, `analysis/protocol_checks.py`, `analysis/protocol_checks2.py`, `analysis/v24_check.py` |

The Kaggle scripts need PyTorch and a GPU; the others need NumPy, SciPy, pandas
and scikit-learn. The analysis scripts are run from `research/src`.

License: MIT
