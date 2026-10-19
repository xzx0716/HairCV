# Scoring System

Score definitions for the image readout. The authoritative rules live in
code: [`../image_analysis/scorer.py`](../image_analysis/scorer.py)
(`ScoreMapper`).

- Every indicator is reported on a **0-10 scale** so the AHP comprehensive
  scoring model can consume the values directly.
- `to_json()` emits a human-readable report; `to_ahp_format()` emits the
  flat indicator-score dict for downstream aggregation.
- Cuticle integrity: grade 0-3 -> 10/7/4/1 (unknown grade -> 0)
- Split-end rate: <5% -> 10, 5-15% -> 7, 15-30% -> 4, >=30% -> 1
- Frizziness: grade 0-3 -> 10/7/4/1
- Shininess: grade 0-2 -> 2/6/10
