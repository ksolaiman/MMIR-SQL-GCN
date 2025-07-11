# prcurve folder contents

This folder stores all precision-recall (P-R) curve data, for different experiments and models.

---

## 📂 /epoch_41/

Results generated using the **epoch_41** trained model variant.

- Contains **11-point** interpolated P-R curve data.
- File naming convention indicates the query → target setting, e.g.:
  - `145-539-precision-11-point.pkl`
  - `145-539-recall-11-point.pkl`
- Subfolders:
  - `/pr_curves/png/` and `/pr_curves/pdf/` store generated plots (PNG, PDF).

Use this folder to evaluate epoch_41 model performance specifically.

---

## 📂 /sim-as-prediction/

These are the P-R curves and results **prepared for the arXiv released paper**.

- Also use **11-point** interpolation (same recall grid).
- Naming format matches epoch_41:
  - `539-145-precision-11-point.pkl`, etc.
- Represents *finalized* or *canonical* curves for publication.

Note:
> You may want to check if these results and **epoch_41** runs are from identical models (to confirm consistency).

- The recall arrays are identical by design—they use the same fixed interpolation grid (e.g. 11-point steps from 0.0 to 1.0).

- The precision arrays can show very small differences even for the same model, because precision depends on ranking outputs, which may vary slightly between runs.

- These differences are normal (often ~0.01) and reflect small changes in ranking, not major model differences.

If comparing /sim-as-prediction/ to /epoch_41/, they might have small variations in precision — but they likely use the same underlying model or very similar checkpoints.

---

## 📂 /base/

Contains the original P-R curves downloaded from the Oct 2022 version of this repo.

- These are **11-point interpolated curves** (recall grid at 0.0 to 1.0 in 0.1 steps).
- Original reference curves from the early days, no longer needed.

---

## 📂 /EARS/

Includes **comparison baseline** runs:

- Contains 11-point P-R curves for the EARS baseline system.
- Structure:
  - `/main-11-point/` folder has recall and precision .pkl files.
  - Older 1000-point versions stored in `/older-1000-points/`, no longer needed but preserved for record.

These can be used to directly compare our model with EARS in the same plot.

---

## General Notes:

- All **recall** .pkl files contain *fixed grid* recall levels (typically [0.0, 0.1, ... 1.0]).
- All **precision** .pkl files contain *interpolated precision values* at those recall grid points.
- Differences in precision arrays reflect actual model performance differences.

---

## 📌 Recommendations:

- Use **/sim-as-prediction/** for *paper-ready* PR curves.
- Use **/epoch_41/** for experimental runs with the epoch_41 checkpoint.
- Use **/EARS/** for baseline comparisons.
- Older or extra folders not listed here are generally not needed.

