# Conflict Calibration (Day-6) — why the first matrix was null, and the fix

## What the first matrix showed
The 9-run portfolio matrix (GRPO/GDPO/CDPO × 3 seeds) produced NO separation —
GRPO actually finished ahead. Diagnosis from the trajectory CSVs: the soft-pref
score (SPS) was ~identical across methods and the measured correlation between
ESG (soft) and hard-constraint satisfaction was ~0. **There was no hard/soft
conflict**, so CDPO's gate — its entire reason to exist — had nothing to
arbitrate, and its extra machinery just made it a slower GRPO.

Root cause in the data generator: `banned_sectors` were drawn only from
OFF-universe sectors (the line `banned -= set(params["sectors"])`), so the H3
"no banned sector" constraint never touched the 4 tradable assets. And ESG
ratings aligned with returns. Maximizing ESG cost nothing on any hard
constraint. Verified empirically: corr(ESG, hard_pass) = -0.007.

## The fix
Two coupled changes:

1. **`conflict_level` in DatasetConfig / `--conflict-level` in generate_full.py.**
   When > 0, the sampler bans the sector of the HIGHEST-ESG asset (scaled by
   tier: easy=0, medium=0.5, hard=1.0). Now the most ESG-rich holding is exactly
   the one H3 forbids.

2. **H3 is now weight-based, not presence-based.** Previously H3 failed if a
   banned sector merely appeared in the plan — which, once a tradable sector is
   banned, is unsatisfiable (all 4 assets are always listed). Now H3 checks that
   *weight* in banned sectors ≤ `banned_weight_tol` (default 0.05). So the model
   CAN pass H3 — by zeroing the banned high-ESG asset — at the cost of ESG.
   That tradeoff is the conflict.

Verified on generated hard-tier data (conflict_level=1.0):
- H3-pass rate ≈ 0.14 (hard but satisfiable, not impossible)
- mean ESG | H3 failed = 0.665  vs  mean ESG | H3 passed = 0.597
- => satisfying the hard constraint costs ~0.07 ESG: a real hard/soft conflict.

## The experimental design (two matrices, one story)
- **conflict_level=0.0** (already run): CDPO ≈ GRPO, no separation. This is now a
  legitimate ABLATION showing the effect is conflict-dependent.
- **conflict_level=1.0** (to run): the regime where CDPO's gate should prevent
  the soft channel from corrupting the hard-constraint gradient. If CDPO's CCR
  curve separates upward here, that IS the paper's result — and the contrast with
  the null ablation is stronger evidence than a single win.

## Commands
```
# regenerate WITH conflict
python scripts/generate_full.py --output data/full_conflict --conflict-level 1.0
python scripts/generate_descriptions.py --data-dir data/full_conflict

# run the matrix on the conflict data (point run_matrix at the new dir)
DATA_DIR=data/full_conflict bash scripts/run_matrix.sh
```

## Verify before you train: measure_conflict.py
Always sanity-check a dataset BEFORE launching the GPU matrix:
```
# measure a generated dataset (exit 0 = PASS/OK to train, 1 = FAIL/don't)
python scripts/measure_conflict.py --data-dir data/full_conflict

# or probe the sampler directly at a level
python scripts/measure_conflict.py --conflict-level 1.0   # expect PASS
python scripts/measure_conflict.py --conflict-level 0.0   # expect FAIL (null)
```
It reports, per difficulty tier: H3-pass rate, and the conflict gap
(mean ESG when failing H3 minus mean ESG when passing). A hard-tier gap ≥ 0.03
means satisfying the hard constraint costs ESG — real conflict — and prints
PASS. The tier gradient (easy≈0 → hard≈0.07) is itself a paper result:
CDPO's advantage should scale with it.
