# -*- coding: utf-8 -*-
"""
Scalp Health AI Analysis - Railway deployment entry point.

Self-contained deployment:
  - Keras multi-attribute scalp model (model/best_model.keras, 24 MB)
  - Classic CV hair-fibre metrics (cv/ package)
  - Upload / Camera analysis + Batch analysis with CSV export

Local test:  python app.py
Railway:     started with `python app.py`; PORT is injected by Railway.
"""

import os
import io
import csv
import time
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "best_model.keras")
IMG_SIZE = 224

ATTR_KEYS = ["oily", "dandruff", "dry", "redness_or_sensitive", "hair_density_issue"]
ATTR_EN = {
    "oily": "Oiliness",
    "dandruff": "Dandruff",
    "dry": "Dryness",
    "redness_or_sensitive": "Redness/Sensitivity",
    "hair_density_issue": "Hair Density Issue",
}

_ENGINE = None


# =============================================================
# Inference engine
# =============================================================
class Engine:
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        self.tf = tf
        print("Loading scalp model...")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        from profile_generator import ProfileGenerator
        self.generator = ProfileGenerator()

        from cv.split_end_detector import SplitEndDetector
        from cv.frizziness_detector import FrizzinessDetector
        from cv.shininess_detector import ShininessDetector
        import cv2
        self.cv2 = cv2
        self.split_detector = SplitEndDetector()
        self.frizz_detector = FrizzinessDetector()
        self.shine_detector = ShininessDetector()
        print("Engine ready.")

    def predict_scalp(self, image: Image.Image) -> dict:
        img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)  # model has built-in Rescaling(1/255)
        x = np.expand_dims(arr, 0)
        logits = np.array(self.model.predict(x, verbose=0)).flatten().tolist()
        return self.generator.generate_from_logits(logits)

    def predict_hair_cv(self, image: Image.Image) -> dict:
        rgb = np.array(image.convert("RGB"))
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        out = {}
        try:
            rate, info = self.split_detector.detect(bgr)
            out["split_end"] = {"value": round(float(rate), 1),
                                "detail": f"{info['split_end_count']} split points / {info['total_hairs']} hairs"}
        except Exception:
            out["split_end"] = {}
        try:
            grade, info = self.frizz_detector.detect(bgr)
            names = {0: "Smooth", 1: "Slightly frizzy", 2: "Moderately frizzy", 3: "Severely frizzy"}
            out["frizziness"] = {"grade": int(grade), "level": names.get(int(grade), "?")}
        except Exception:
            out["frizziness"] = {}
        try:
            grade, info = self.shine_detector.detect(bgr)
            names = {0: "Dull", 1: "Medium shine", 2: "Shiny"}
            out["shininess"] = {"grade": int(grade), "level": names.get(int(grade), "?")}
        except Exception:
            out["shininess"] = {}
        return out

    def analyze(self, image: Image.Image) -> dict:
        return {"scalp": self.predict_scalp(image), "hair_cv": self.predict_hair_cv(image)}


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


# =============================================================
# Visualisation & report
# =============================================================
def plot_radar(scores: dict) -> Image.Image:
    labels = [ATTR_EN[k] for k in ATTR_KEYS]
    values = [scores[k] for k in ATTR_KEYS]
    angles = np.linspace(0, 2 * np.pi, len(ATTR_KEYS), endpoint=False).tolist()
    v_loop = values + values[:1]
    a_loop = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw=dict(polar=True))
    ax.fill(a_loop, v_loop, color="#4C8BF5", alpha=0.25)
    ax.plot(a_loop, v_loop, color="#4C8BF5", linewidth=2)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
    ax.set_title("Scalp 5-Dimension Tendency", fontsize=14, pad=18)
    for ang, val in zip(angles, values):
        ax.text(ang, val + 6, str(val), ha="center", fontsize=10,
                fontweight="bold", color="#1a56c4")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _level(v):
    return "🔴 Strong" if v >= 60 else ("🟡 Mild" if v >= 30 else "🟢 None")


def _summary(scores: dict, healthy: int) -> str:
    strong = [ATTR_EN[k] for k in ATTR_KEYS if scores[k] >= 60]
    mild = [ATTR_EN[k] for k in ATTR_KEYS if 30 <= scores[k] < 60]
    none = [ATTR_EN[k] for k in ATTR_KEYS if scores[k] < 30]
    parts = []
    if strong:
        parts.append(f"strong tendency: {', '.join(strong)}")
    if mild:
        parts.append(f"mild tendency: {', '.join(mild)}")
    if none:
        parts.append(f"no obvious signs of: {', '.join(none)}")
    body = "; ".join(parts) if parts else "No clear tendency detected."
    if healthy >= 70:
        verdict = "Overall scalp condition appears good."
    elif healthy >= 40:
        verdict = "Overall condition is moderate; targeted care is recommended."
    else:
        verdict = "Multiple concerns detected; professional consultation is advised."
    return f"{verdict} Detected {body}."


def build_report(result: dict) -> str:
    scores = result["scalp"]["percentage_scores"]
    healthy = result["scalp"]["healthy"]["healthy"]
    hc = result["hair_cv"]
    hc_icon = "🟢" if healthy >= 70 else ("🟡" if healthy >= 40 else "🔴")
    md = f"## {hc_icon} Overall Health Score: {healthy}/100\n\n"
    md += f"> {_summary(scores, healthy)}\n\n"
    md += "### Scalp 5-Dimension Tendency Scores\n\n"
    md += "| Attribute | Score | Level |\n|---|---|---|\n"
    for k in ATTR_KEYS:
        v = scores[k]
        bar = "█" * int(v / 5) + "░" * (20 - int(v / 5))
        md += f"| {ATTR_EN[k]} | **{v}**/100 | {_level(v)} `{bar}` |\n"
    md += "\n### Hair Fibre Metrics (Classic CV)\n\n"
    md += "| Metric | Result | Notes |\n|---|---|---|\n"
    se = hc.get("split_end", {})
    fz = hc.get("frizziness", {})
    sh = hc.get("shininess", {})
    md += (f"| Split-end rate | {se['value']}% | {se.get('detail', '')} |\n"
           if "value" in se else "| Split-end rate | N/A | - |\n")
    md += (f"| Frizziness | {fz.get('level', '?')} (level {fz.get('grade', '?')}) | 0=smooth, 3=severe |\n"
           if "grade" in fz else "| Frizziness | N/A | - |\n")
    md += (f"| Shine | {sh.get('level', '?')} (level {sh.get('grade', '?')}) | 0=dull, 2=shiny |\n"
           if "grade" in sh else "| Shine | N/A | - |\n")
    md += ("\n---\n⚠️ Results are AI visual tendency analysis for reference only; "
           "not a medical diagnosis.")
    return md


# =============================================================
# Callbacks
# =============================================================
def analyze_photo(image):
    if image is None:
        return None, "❌ Please upload or take a scalp photo first."
    try:
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        eng = get_engine()
        result = eng.analyze(pil)
        return plot_radar(result["scalp"]["percentage_scores"]), build_report(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Analysis failed: {str(e)[:200]}"


def batch_analyze(files, progress=gr.Progress()):
    if not files:
        return None, None
    eng = get_engine()
    rows = []
    for f in progress.tqdm(files, desc="Analysing..."):
        fname = os.path.basename(f)
        try:
            pil = Image.open(f).convert("RGB")
            r = eng.analyze(pil)
            s, h, hc = r["scalp"]["percentage_scores"], r["scalp"]["healthy"]["healthy"], r["hair_cv"]
            rows.append({
                "File": fname, "HealthScore": h,
                "Oiliness": s["oily"], "Dandruff": s["dandruff"], "Dryness": s["dry"],
                "Redness/Sensitive": s["redness_or_sensitive"], "DensityIssue": s["hair_density_issue"],
                "SplitEndRate%": hc.get("split_end", {}).get("value", ""),
                "FrizzLevel": hc.get("frizziness", {}).get("grade", ""),
                "ShineLevel": hc.get("shininess", {}).get("grade", ""),
            })
        except Exception as e:
            rows.append({"File": fname, "HealthScore": f"failed:{str(e)[:30]}"})
        time.sleep(0.05)

    csv_path = os.path.join(BASE, f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
    except Exception:
        df = rows
    return df, csv_path


# =============================================================
# UI
# =============================================================
def build_ui():
    with gr.Blocks(title="Scalp Health AI Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🧬 Scalp Health AI Analysis\n"
            "Upload or take a **scalp / hair photo** to get 5-dimension scalp tendency scores "
            "(oiliness / dandruff / dryness / redness-sensitivity / hair density), an overall health score, "
            "plus split-end rate, frizziness and shine metrics.\n\n"
            "**Phone workflow with a MaxSee scanner**: take photos in MaxSee (saved to gallery) → "
            "open this page → upload the gallery photos."
        )
        with gr.Tabs():
            with gr.Tab("📸 Upload / Camera"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(label="Scalp photo (upload or camera)", type="numpy",
                                          sources=["upload", "webcam"], height=360)
                        btn = gr.Button("🔍 Analyse", variant="primary", size="lg")
                        gr.Markdown("<small>Tips: part the hair to expose the scalp under natural light, "
                                    "shoot 10–20 cm away, keep the image sharp.</small>")
                    with gr.Column():
                        radar_out = gr.Image(label="5-dimension radar chart", height=400)
                        report_out = gr.Markdown()
                btn.click(analyze_photo, inputs=img_in, outputs=[radar_out, report_out])

            with gr.Tab("📊 Batch Analysis"):
                gr.Markdown("Upload multiple scalp photos; each is analysed and all results export as a CSV table.")
                files_in = gr.File(label="Select multiple images", file_count="multiple",
                                   file_types=["image"], height=180)
                bbtn = gr.Button("📊 Start Batch Analysis", variant="primary")
                table_out = gr.Dataframe(label="Results", wrap=True)
                csv_out = gr.File(label="Download CSV")
                bbtn.click(batch_analyze, inputs=files_in, outputs=[table_out, csv_out])

        gr.Markdown(
            "---\n**Model info**: scalp 5-attribute model = EfficientNetV2-B0 multi-label network "
            "(top-3 attribute recall ≈ 79%); fibre metrics use classic computer vision. "
            "All outputs are AI visual tendency analysis for reference only — **not a medical diagnosis**."
        )
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    get_engine()
    build_ui().launch(server_name="0.0.0.0", server_port=port)
