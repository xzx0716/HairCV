# -*- coding: utf-8 -*-
"""
Scalp Health AI Analysis App (Gradio Web UI, English)

Three modes:
  1. Upload / Camera: pick images from phone gallery or computer
  2. Direct Device Capture: computer connects to the scanner's WiFi hotspot,
     captures a frame and analyses it (no phone needed)
  3. Batch Analysis: upload many images, export a CSV table

Start:
  python -m app.scalp_app            # LAN access (phone on the same WiFi)
  python -m app.scalp_app --share    # public link (works on mobile data / other networks)

Network notes:
  - LAN link (http://<computer-IP>:7860) requires phone and computer on the SAME network
  - Public link (--share) works from anywhere, but the computer needs internet
  - Typical flow with a MaxSee-style scanner:
      a) Phone connects to scanner hotspot -> take photos (saved to gallery)
      b) Phone switches back to the shop/campus WiFi
      c) Phone opens the app link and uploads the gallery photos
    OR: computer connects scanner hotspot and uses the "Direct Device Capture" tab.
"""

import os
import sys
import io
import csv
import time
import argparse
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr
from app.unified_engine import UnifiedScalpEngine
from app.device_capture import discover_device, grab_frame

_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = UnifiedScalpEngine(enable_torch=False, enable_sam2=True)
    return _ENGINE


# English attribute metadata
ATTR_KEYS = ["oily", "dandruff", "dry", "redness_or_sensitive", "hair_density_issue"]
ATTR_EN = {
    "oily": "Oiliness",
    "dandruff": "Dandruff",
    "dry": "Dryness",
    "redness_or_sensitive": "Redness/Sensitivity",
    "hair_density_issue": "Hair Density Issue",
}


def _level_en(v):
    """Score (0-100) -> English level label"""
    if v >= 60:
        return "🔴 Strong"
    if v >= 30:
        return "🟡 Mild"
    return "🟢 None"


# ---------------------------------------------------------------
# Visualisation & report
# ---------------------------------------------------------------
def plot_scalp_radar(scores: dict) -> Image.Image:
    labels = [ATTR_EN[k] for k in ATTR_KEYS]
    values = [scores[k] for k in ATTR_KEYS]
    angles = np.linspace(0, 2 * np.pi, len(ATTR_KEYS), endpoint=False).tolist()
    values_loop = values + values[:1]
    angles_loop = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw=dict(polar=True))
    ax.fill(angles_loop, values_loop, color="#4C8BF5", alpha=0.25)
    ax.plot(angles_loop, values_loop, color="#4C8BF5", linewidth=2)
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


def _english_summary(scores: dict, healthy: int) -> str:
    """Build an English interpretation from the score values"""
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


def build_markdown_report(result: dict) -> str:
    scalp = result["scalp"]
    scores = scalp["percentage_scores"]
    healthy = scalp["healthy"]["healthy"]
    hair_cv = result["hair_cv"]

    hc = "🟢" if healthy >= 70 else ("🟡" if healthy >= 40 else "🔴")
    md = f"## {hc} Overall Health Score: {healthy}/100\n\n"
    md += f"> {_english_summary(scores, healthy)}\n\n"

    md += "### Scalp 5-Dimension Tendency Scores\n\n"
    md += "| Attribute | Score | Level |\n|---|---|---|\n"
    for k in ATTR_KEYS:
        v = scores[k]
        name = ATTR_EN[k]
        bar = "█" * int(v / 5) + "░" * (20 - int(v / 5))
        md += f"| {name} | **{v}**/100 | {_level_en(v)} `{bar}` |\n"
    md += "\n"

    md += "### Hair Fibre Metrics (Classic CV)\n\n"
    se = hair_cv.get("split_end", {})
    fz = hair_cv.get("frizziness", {})
    sh = hair_cv.get("shininess", {})
    md += "| Metric | Result | Notes |\n|---|---|---|\n"
    if "value" in se:
        md += f"| Split-end rate | {se['value']}% | {se.get('detail', '')} |\n"
    else:
        md += "| Split-end rate | N/A | - |\n"
    fz_names = {0: "Smooth", 1: "Slightly frizzy", 2: "Moderately frizzy", 3: "Severely frizzy"}
    if "level" in fz:
        md += f"| Frizziness | {fz_names.get(int(fz['grade']), '?')} (level {fz['grade']}) | 0=smooth, 3=severe |\n"
    else:
        md += "| Frizziness | N/A | - |\n"
    sh_names = {0: "Dull", 1: "Medium shine", 2: "Shiny"}
    if "level" in sh:
        md += f"| Shine | {sh_names.get(int(sh['grade']), '?')} (level {sh['grade']}) | 0=dull, 2=shiny |\n"
    else:
        md += "| Shine | N/A | - |\n"

    if "damage" in result and isinstance(result["damage"], dict) and "level" in result["damage"]:
        d = result["damage"]
        lvl_en = {0: "Healthy", 1: "Mild damage", 2: "Severe damage"}.get(d["level"], d["level"])
        md += f"\n### Hair Damage Assessment (Deep Learning)\n\n- Damage level: **{lvl_en}** (confidence {d['confidence']}%)\n"

    md += f"\n---\n⚠️ {result.get('disclaimer', 'Results are AI visual tendency analysis for reference only; not a medical diagnosis.')}\n"
    md += f"\nℹ️ {result.get('note', '')}\n"
    return md


def _run_analysis(pil_img: Image.Image, use_sam2: bool = False):
    engine = get_engine()
    result = engine.analyze(pil_img, use_sam2=bool(use_sam2))
    radar = plot_scalp_radar(result["scalp"]["percentage_scores"])
    report = build_markdown_report(result)
    return radar, report, result


# ---------------------------------------------------------------
# Tab 1: upload / camera
# ---------------------------------------------------------------
def analyze_photo(image, use_sam2):
    if image is None:
        return None, "❌ Please upload or take a scalp photo first."
    try:
        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        radar, report, _ = _run_analysis(pil_img, use_sam2=use_sam2)
        return radar, report
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Analysis failed: {str(e)[:200]}"


# ---------------------------------------------------------------
# Tab 2: direct device capture
# ---------------------------------------------------------------
def device_capture_and_analyze(device_url):
    """Connect to scanner -> grab frame -> analyse (device frames are close-ups, no SAM2 needed)"""
    url = device_url.strip() if device_url and device_url.strip() else None
    frame, msg = grab_frame(url)
    if frame is None:
        return None, None, (
            f"❌ {msg}\n\n**Troubleshooting**:\n"
            "1. Connect this computer to the scanner's WiFi hotspot (same hotspot your phone uses in MaxSee).\n"
            "2. Confirm MaxSee shows live video on your phone first.\n"
            "3. Click \"Capture & Analyse\" again.\n"
            "4. If this computer cannot join the hotspot, use the phone workflow instead:\n"
            "   take photos in MaxSee (saved to gallery) → switch phone back to the normal WiFi → "
            "open this app and upload the photos in the **Upload / Camera** tab."
        )
    try:
        pil_img = Image.fromarray(frame)
        radar, report, _ = _run_analysis(pil_img, use_sam2=False)
        return frame, radar, f"✅ {msg}\n\n{report}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return frame, None, f"⚠️ Capture succeeded but analysis failed: {str(e)[:200]}"


def device_probe():
    base = discover_device(verbose=True)
    if base:
        return f"✅ Scanner found: `{base}` — click **Capture & Analyse**."
    return ("❌ Scanner not found. Make sure this computer is connected to the scanner's WiFi hotspot. "
            "If your computer cannot join that hotspot, use the phone+upload workflow described above.")


# ---------------------------------------------------------------
# Tab 3: batch
# ---------------------------------------------------------------
def batch_analyze(files, use_sam2, progress=gr.Progress()):
    if not files:
        return None, None
    engine = get_engine()
    rows = []
    for f in progress.tqdm(files, desc="Analysing..."):
        fname = os.path.basename(f)
        try:
            pil_img = Image.open(f).convert("RGB")
            result = engine.analyze(pil_img, use_sam2=bool(use_sam2))
            s = result["scalp"]["percentage_scores"]
            h = result["scalp"]["healthy"]["healthy"]
            hc = result["hair_cv"]
            rows.append({
                "File": fname,
                "HealthScore": h,
                "Oiliness": s["oily"],
                "Dandruff": s["dandruff"],
                "Dryness": s["dry"],
                "Redness/Sensitive": s["redness_or_sensitive"],
                "DensityIssue": s["hair_density_issue"],
                "SplitEndRate%": hc.get("split_end", {}).get("value", ""),
                "FrizzLevel": hc.get("frizziness", {}).get("grade", ""),
                "ShineLevel": hc.get("shininess", {}).get("grade", ""),
            })
        except Exception as e:
            rows.append({"File": fname, "HealthScore": f"failed:{str(e)[:30]}"})
        time.sleep(0.05)

    csv_path = os.path.join(ROOT, f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
    except Exception:
        df = rows
    return df, csv_path


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Scalp Health AI Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🧬 Scalp Health AI Analysis\n"
            "Three modes: **Upload / Camera**, **Direct Scanner Capture (WiFi)**, and **Batch Analysis (CSV export)**.\n"
            "Outputs: 5-dimension scalp tendency scores (oiliness / dandruff / dryness / redness-sensitivity / hair density), "
            "overall health score, split-end rate, frizziness and shine.\n\n"
            "**Using a phone with a MaxSee scanner?** Take photos in MaxSee (they save to your gallery) → "
            "switch the phone back to the normal WiFi → open this page and upload the photos. "
            "Or connect your *computer* to the scanner hotspot and use the Direct Capture tab."
        )

        with gr.Tabs():
            # ---------------- Tab 1 ----------------
            with gr.Tab("📸 Upload / Camera"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(label="Scalp photo (upload or camera)", type="numpy",
                                          sources=["upload", "webcam"], height=360)
                        sam2_ck = gr.Checkbox(label="Smart ROI crop (SAM2 background removal — advised for selfies)", value=False)
                        btn = gr.Button("🔍 Analyse", variant="primary", size="lg")
                        gr.Markdown("<small>Tips: part the hair to expose the scalp under natural light, "
                                    "shoot 10–20 cm away, keep the image sharp.</small>")
                    with gr.Column():
                        radar_out = gr.Image(label="5-dimension radar chart", height=400)
                        report_out = gr.Markdown()
                btn.click(analyze_photo, inputs=[img_in, sam2_ck],
                          outputs=[radar_out, report_out])

            # ---------------- Tab 2 ----------------
            with gr.Tab("🔭 Direct Scanner Capture"):
                gr.Markdown(
                    "**No phone needed**: connect *this computer* to the scanner's WiFi hotspot "
                    "(the same hotspot your phone uses for MaxSee), then capture a frame directly.\n\n"
                    "Steps: ① Connect computer to the scanner hotspot → ② Click **Probe Device** → "
                    "③ Click **Capture & Analyse**.\n\n"
                    "⚠️ If your computer cannot join the scanner hotspot, use the phone workflow: "
                    "shoot in MaxSee → switch phone back to normal WiFi → upload photos in the first tab."
                )
                with gr.Row():
                    dev_url = gr.Textbox(
                        label="Device address (leave empty for auto-detect)",
                        placeholder="http://192.168.10.123:8080",
                        scale=3,
                    )
                    probe_btn = gr.Button("🔌 Probe Device", scale=1)
                probe_status = gr.Markdown()
                cap_btn = gr.Button("📸 Capture & Analyse", variant="primary", size="lg")
                with gr.Row():
                    dev_frame = gr.Image(label="Captured frame", height=380)
                    dev_radar = gr.Image(label="5-dimension radar chart", height=380)
                dev_report = gr.Markdown()

                probe_btn.click(device_probe, outputs=probe_status)
                cap_btn.click(device_capture_and_analyze, inputs=dev_url,
                              outputs=[dev_frame, dev_radar, dev_report])

            # ---------------- Tab 3 ----------------
            with gr.Tab("📊 Batch Analysis"):
                gr.Markdown("Upload multiple scalp photos; each is analysed and all results are exported as a CSV table.")
                files_in = gr.File(label="Select multiple images", file_count="multiple",
                                   file_types=["image"], height=180)
                bsam2 = gr.Checkbox(label="Smart ROI crop", value=False)
                bbtn = gr.Button("📊 Start Batch Analysis", variant="primary")
                table_out = gr.Dataframe(label="Results", wrap=True)
                csv_out = gr.File(label="Download CSV")
                bbtn.click(batch_analyze, inputs=[files_in, bsam2],
                           outputs=[table_out, csv_out])

        gr.Markdown(
            "---\n"
            "**Model info**: scalp 5-attribute model = EfficientNetV2-B0 multi-label network "
            "(top-3 attribute recall ≈ 79%); fibre metrics use classic computer vision; "
            "direct capture supports MaxSee/Inskam-type WiFi scanners (standard MJPEG/snapshot protocol). "
            "All outputs are AI visual tendency analysis for reference only — **not a medical diagnosis**."
        )
    return demo


def _print_lan_urls(port: int):
    """Print all local IPs of this machine (the IP changes when switching networks)"""
    import socket
    print("\n" + "=" * 60)
    print("Access addresses (phone must be on the SAME network as this computer):")
    print(f"   This computer:  http://localhost:{port}")
    try:
        _, _, ips = socket.gethostbyname_ex(socket.gethostname())
        for ip in ips:
            print(f"   Other devices:  http://{ip}:{port}")
    except Exception:
        pass
    print("Use the IP that belongs to the network both devices share.")
    print("For mobile data / different networks: restart with --share for a public link.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Scalp Health AI Analysis App")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("=" * 60)
    print("Scalp Health AI Analysis - starting (first run loads models, please wait)...")
    print("=" * 60)
    get_engine()
    _print_lan_urls(args.port)
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
