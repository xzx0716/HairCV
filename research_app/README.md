# Research App

Three-tab research Gradio application (superset of the production
`../web_app`):

1. **Upload / Camera analysis** — single-image inference.
2. **Batch analysis** — table with CSV export.
3. **Device direct connection** — prototype tab for live capture from a
   scalp-detection device on the same network.

## Files

| File | Purpose |
| --- | --- |
| `scalp_app.py` | Gradio UI, three tabs (English interface) |
| `unified_engine.py` | Unified inference engine joining the neural scalp model and the classic CV detectors |
| `device_capture.py` | Device direct-connection capture helper |

This app is the research version; only the two production tabs are
deployed on Railway. The device tab requires same-network access and is
kept here for hardware integration experiments.
