# Web App (production)

The self-contained Gradio application deployed on Railway at
<https://scalp-health-ai-production.up.railway.app>. English UI, two tabs:

1. **Upload / Camera analysis** — single image inference with radar chart
   and detailed score report.
2. **Batch analysis** — multi-image table with CSV export.

## Contents

```text
web_app/
|-- app.py                 Gradio UI + inference wiring (binds 0.0.0.0:$PORT)
|-- profile_generator.py   Sigmoid x100 -> 0-100 scores, radar chart, report
|-- cv/                    Three fibre detectors (split end, frizz, shine)
|-- model/best_model.keras Trained EfficientNetV2-B0 scalp model
|-- requirements.txt       Pinned stack (gradio 6.17.3, tensorflow-cpu 2.17.1)
|-- railway.json           Start command for Railway (python app.py)
`-- .python-version        Python 3.11 pin
```

## Run locally

```bash
pip install -r requirements.txt
python app.py        # http://localhost:7860
```

## Deploy to Railway

```bash
cd hair_benchmark/web_app
railway up --service scalp-health-ai --detach
```

Railway injects `PORT`; the app reads it and binds `0.0.0.0`. Do not add a
healthcheck path — Gradio has no `/health` endpoint.
