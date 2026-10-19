# Hair Benchmark

Production code and reference tooling for the BioWipe image readout: two
complementary analysis paths that turn scalp and hair photographs into
structured 0-10 indicator scores.

## Two analysis paths

1. **Neural scalp model** — an EfficientNetV2-B0 network (6,067,989 params,
   621,381 trainable) with five independent sigmoid outputs (oily, dandruff,
   dryness, redness/sensitivity, hair-density issue). Trained on 1,931
   Roboflow images with partial (weak) labels; on the held-out 57-image test
   split the top-1 attribute recall is 42.1% and top-3 recall is 78.9%.
2. **Classic CV fibre detectors** — rule-based OpenCV pipelines for
   cuticle integrity, split-end rate, frizziness and shininess. They assume
   dark strands on a light background and are not interpretable on
   scalp-surface macro images.

Scores are reported on a 0-10 scale per indicator so the AHP comprehensive
scoring model can consume them directly.

## Layout

```text
hair_benchmark/
|-- image_analysis/   Reusable CV library: preprocessing, four fibre
|                     detectors, score mapping (0-10, AHP format)
|-- web_app/          Self-contained production Gradio app (deployed on
|                     Railway), including the trained Keras weights
|-- research_app/     Three-tab research Gradio app (upload/camera, batch,
|                     device direct-connection prototype)
|-- wiki_figures/     Script that regenerates every figure on the iGEM wiki
|                     Detection page from real project artifacts
|-- scoring_system/   Score definitions overview (rules live in
|                     image_analysis/scorer.py)
|-- sample_images/    Small sample references
`-- tests/            Pytest suite for the image_analysis library
```

## Quick start

```bash
# Library tests (20 unit tests, no model download needed)
pip install pytest numpy opencv-python-headless
pytest hair_benchmark/tests/test_haircv.py -v

# Production web app (needs Python 3.11 for TensorFlow 2.17)
cd hair_benchmark/web_app
pip install -r requirements.txt
python app.py            # binds 0.0.0.0:7860 by default
```

The trained scalp-model weights ship in
`web_app/model/best_model.keras` (best checkpoint: epoch 17,
val_loss 0.458).

## Deployment

The `web_app/` folder is a self-contained Railway package
(`railway.json` start command `python app.py`, Python 3.11 pinned via
`.python-version`, server binds `0.0.0.0:$PORT`). The live production
instance runs at <https://scalp-health-ai-production.up.railway.app>.

## Known limits

- Fibre detectors assume dark-strands-on-light-background input; their
  outputs on scalp-surface macro frames are not interpretable (input-type
  gating is future work).
- SAM2 segmentation is used only for macroscopic hair photos; it is not
  suitable for FDU microscopic scalp frames.
- CV indicators are image-measurable proxies only; tensile strength, wet
  combing resistance and moisture content stay with Wet Lab.
