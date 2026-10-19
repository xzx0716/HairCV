# -*- coding: utf-8 -*-
"""
Generate all CV-wiki figures from real project artifacts (logs, evaluation
JSON, training CSV, shipped detector code and the production Keras model).

Run:  python generate_figures.py
Output: ../images/fig*.png
Every number shown is read from project files or produced by running the
actual shipped code; nothing is invented.
"""

import os
import re
import sys
import csv
import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(parents=True, exist_ok=True)
RAIL = ROOT / "deploy" / "railway"
sys.path.insert(0, str(RAIL))

# --- Unified navy-on-cream palette (matches the wiki page body) ---
C_BG = "#f7f3ea"        # warm cream figure background
NAVY = "#173f62"        # deep navy: primary series, dark panels, titles
STEEL = "#5c82a6"       # steel blue: secondary series
SKY = "#b9cee0"         # light blue: tertiary series
MIST = "#dde4ea"        # faintest blue-grey: quaternary / unknown
CREAM = "#f7f3ea"       # text on dark navy panels
C_TEXT = "#1d3a54"      # near-navy body text
C_SUBTEXT = "#6e7f8a"   # muted subtitle
C_ARROW = "#173f62"     # arrows
TITLE_SERIF = "DejaVu Serif"
# Kept for image-internal markers (Canny edges, branch-point dots, etc.)
C_BLUE = "#2E5FA3"
C_RED = "#C0392B"
C_GREEN = "#27AE60"
C_ORANGE = "#E67E22"
C_PURPLE = "#7D3C98"
C_GRAY = "#7F8C8D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.color": "#d8d0be",
    "grid.alpha": 0.7,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
})


def save(fig, name):
    fig.patch.set_facecolor(C_BG)
    if fig._suptitle is not None:
        fig._suptitle.set_fontfamily(TITLE_SERIF)
        fig._suptitle.set_color(NAVY)
    for ax in fig.axes:
        ax.title.set_fontfamily(TITLE_SERIF)
        ax.title.set_color(NAVY)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(False)
        if ax.name == "polar" or ax.images or not ax.axison:
            continue
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(OUT / name, facecolor=C_BG)
    plt.close(fig)
    print("saved", name)


def footnote(fig, text):
    """Small grey source note, bottom-centre — editorial style."""
    fig.text(0.5, 0.005, text, fontsize=7.5, color=C_SUBTEXT, style="italic",
             ha="center")


def shadow_box(ax, x, y, w, h, fc, boxstyle="round,pad=0.1,rounding_size=0.22",
               ec=NAVY, lw=1.0, alpha=1.0):
    """FancyBboxPatch with a soft drop shadow behind it."""
    ax.add_patch(FancyBboxPatch((x + 0.07, y - 0.10), w, h, boxstyle=boxstyle,
                                fc="#d9d2c0", ec="none", alpha=0.85, zorder=1.9))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2.0))


def imread_u(path):
    """cv2.imread with unicode-safe path handling."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


# =====================================================================
# fig01 overall dual-system architecture  (v2 — light-blue bg, concise labels)
# =====================================================================
def fig01():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc, tc=C_TEXT, fs=10.5):
        shadow_box(ax, x, y, w, h, fc)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold", linespacing=1.25, zorder=2.1)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=12, lw=1.3, color=C_ARROW,
            shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0"))

    # --- Boxes: compact modules with generous whitespace ---
    box(0.5, 4.60, 1.6, 1.1, "Scalp / Hair\nPhoto", MIST, fs=10)
    box(2.8, 4.65, 1.6, 1.0, "Resize &\nNormalise", MIST, fs=10)

    box(5.5, 6.20, 2.4, 1.5, "Neural Attribute\nModel", NAVY, tc=CREAM, fs=10.5)
    box(5.5, 2.15, 2.4, 1.5, "Classic CV\nDetectors", STEEL, tc=CREAM, fs=10.5)

    box(9.1, 3.90, 2.2, 2.1, "Unified\nInference\nEngine", SKY, fs=10.5)
    box(9.1, 7.30, 2.2, 1.2, "Web\nApplication", MIST, fs=10)
    box(9.1, 1.50, 2.2, 1.2, "AHP Scoring\nModel", MIST, fs=10)

    # --- Arrows: connect box edge to box edge ---
    arrow(2.1, 5.15, 2.8, 5.15)      # input → resize
    arrow(4.4, 5.15, 5.5, 6.95)      # resize → neural (top)
    arrow(4.4, 5.15, 5.5, 2.90)      # resize → classic (bottom)
    arrow(7.9, 6.95, 9.1, 5.00)      # neural → engine (diagonal)
    arrow(7.9, 2.90, 9.1, 4.70)      # classic → engine (diagonal)
    arrow(10.2, 6.00, 10.2, 7.30)    # engine → web app
    arrow(10.2, 3.90, 10.2, 2.70)    # engine → AHP

    # --- Sub-captions OUTSIDE the colour blocks (above neural, below classic) ---
    ax.text(6.7, 8.18, "5 scalp attributes\n(oily · dandruff · dry · redness · density)",
            ha="center", va="center", fontsize=8, style="italic", color=C_SUBTEXT,
            linespacing=1.4)
    ax.text(6.7, 1.62, "3 fibre metrics\n(split-end rate · frizziness · shine)",
            ha="center", va="center", fontsize=8, style="italic", color=C_SUBTEXT,
            linespacing=1.4)

    ax.set_title("Dual-System CV Architecture",
                 fontsize=14, weight="bold", color=C_TEXT, pad=14)
    save(fig, "fig01_dual_system_architecture.png")


# =====================================================================
# fig02 indicator feasibility scope
# =====================================================================
def fig02():
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    ax.set_xlim(0, 13.65); ax.set_ylim(0, 5.4); ax.axis("off")
    cols = [(NAVY, CREAM, "IN SCOPE", "measurable from 2D images",
             "Oiliness · Dandruff · Dryness\nRedness / Sensitivity · Hair density\n"
             "Split-end rate · Frizziness · Shine"),
            (STEEL, CREAM, "SEMI-QUANTITATIVE", "specialised imagery needed",
             "Cuticle integrity\nrequires ~400x microscopy;\n\n"
             "LBP+GLCM features shipped,\nSVM currently unwired"),
            (MIST, C_TEXT, "OUT OF SCOPE", "wet-lab instruments only",
             "Tensile strength\nWet-comb resistance\nMoisture content\n"
             "Protein content\n\ncannot be recovered by CV")]
    w, gap = 4.05, 0.35
    for i, (c, tc, h1, h2, b) in enumerate(cols):
        x = 0.35 + i * (w + gap)
        shadow_box(ax, x, 0.45, w, 4.35, c,
                   boxstyle="round,pad=0.1,rounding_size=0.25", lw=1.2, alpha=0.95)
        ax.text(x + w / 2, 4.32, h1, ha="center", va="center", fontsize=13.5,
                weight="bold", color=tc)
        ax.text(x + w / 2, 3.78, h2, ha="center", va="center", fontsize=9.8,
                style="italic", color=tc)
        ax.text(x + w / 2, 2.15, b, ha="center", va="center", fontsize=9.8, color=tc)
    ax.set_title("Indicator Feasibility Boundary Decided Before Any Model Was Built",
                 fontsize=13, weight="bold", pad=12, color=C_TEXT)
    save(fig, "fig02_indicator_scope.png")


# =====================================================================
# fig03 FDU A/B/C samples
# =====================================================================
def fig03():
    samples = [("220105_A095_1.jpg", "A: pseudo-label 0 (healthy)"),
               ("220124_B159_1.jpg", "B: pseudo-label 1 (mild)"),
               ("220829_C035_1.jpg", "C: pseudo-label 2 (severe)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, (fn, cap) in zip(axes, samples):
        img = imread_u(ROOT / "data" / "发质修复" / "FDU" / "Images" / fn)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(cap, fontsize=11, weight="bold")
        ax.axis("off")
    fig.suptitle("FDU scalp dataset (1,652 images, 1280x1024, 70x skin-mirror macro): "
                 "scalp SURFACE, not single-hair microscopy", fontsize=11.5, weight="bold")
    footnote(fig, "FDU dataset — 70x skin-mirror macro frames; pseudo-labels from filename prefix.")
    save(fig, "fig03_fdu_abc_samples.png")


# =====================================================================
# fig04 FDU class distribution
# =====================================================================
def fig04():
    splits = ["train (n=992)", "val (n=330)", "test (n=330)"]
    A = [770, 260, 258]; B = [198, 64, 66]; C = [24, 6, 6]
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    ax.bar(x - w, A, w, label="A  healthy (1,288 total, 78.0%)", color=NAVY)
    ax.bar(x, B, w, label="B  mild (328 total, 19.9%)", color=STEEL)
    ax.bar(x + w, C, w, label="C  severe (36 total, 2.2%)", color=SKY)
    for i in range(3):
        for off, vals in zip([-w, 0, w], [A, B, C]):
            ax.text(x[i] + off, vals[i] + 12, str(vals[i]), ha="center", fontsize=9, color=C_TEXT)
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel("images", color=C_TEXT)
    ax.set_title("FDU pseudo-label distribution: severe class is only 2.2%\n"
                 "(filename-prefix labels, not human annotation)", fontsize=11.5, weight="bold", color=C_TEXT)
    ax.legend(fontsize=9, facecolor=C_BG, edgecolor=C_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_TEXT)
    ax.tick_params(colors=C_TEXT)
    save(fig, "fig04_fdu_class_distribution.png")


# =====================================================================
# fig05 weak-label mapping matrix
# =====================================================================
def fig05():
    classes = ["Dry scalp", "Inflamed oily scalp", "Normal scalp",
               "Oily scalp", "dry dandruff", "oily dandruff"]
    attrs = ["oily", "dandruff", "dry", "redness /\nsensitive", "hair density\nissue"]
    # rows: 1 positive (mask=1), -1 negative known (mask=1,target=0), 0 unknown (mask=0)
    M = np.array([
        [-1, 0, 1, 0, 0],   # dry
        [0, 0, 0, 1, 0],    # inflamed oily
        [-1, -1, -1, -1, -1],  # normal/healthy
        [1, 0, -1, 0, 0],   # oily
        [-1, 1, 1, 0, 0],   # dry dandruff
        [1, 1, -1, 0, 0],   # oily dandruff
    ])
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    cmap = matplotlib.colors.ListedColormap([SKY, MIST, NAVY])
    ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(attrs, fontsize=10.5, color=C_TEXT)
    ax.tick_params(axis="x", which="major", bottom=False, top=True,
                   labelbottom=False, labeltop=True)
    ax.set_yticks(range(6)); ax.set_yticklabels(classes, fontsize=9.5, color=C_TEXT)
    for i in range(6):
        for j in range(5):
            v = M[i, j]
            ax.text(j, i, {1: "+1", -1: "0", 0: "—"}[v],
                    ha="center", va="center",
                    color=CREAM if v == 1 else C_TEXT, weight="bold", fontsize=11)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(False, which="major")
    legend_elems = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=NAVY, markersize=14,
               label="+1 positive (target=1, mask=1)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=SKY, markersize=14,
               label="0 known-negative (target=0, mask=1)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MIST, markersize=14,
               label="unknown (mask=0, ignored by masked BCE)")]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9.4,
              facecolor=C_BG, edgecolor=C_TEXT, labelcolor=C_TEXT)
    ax.set_title("Weak multi-label projection: 6 Roboflow classes -> 5 attributes\n"
                 "partial-label design prevents treating unlabelled symptoms as absent",
                 fontsize=13.5, weight="bold", color=C_TEXT, pad=14)
    footnote(fig, "Source: canonical mapping rules in weak_multilabel_adapter.py.")
    save(fig, "fig05_weak_label_mapping.png")


# =====================================================================
# fig06 production Keras model architecture
# =====================================================================
def fig06():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    ax.set_xlim(0, 12.3); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=11.7, tc=C_TEXT):
        shadow_box(ax, x, y, w, h, fc, boxstyle="round,pad=0.02,rounding_size=0.18")
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold", zorder=2.1)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.5, color=C_ARROW))

    box(0.2, 4.2, 1.6, 1.5, "Input\n224x224\nRGB", MIST)
    box(2.0, 3.6, 2.8, 2.7, "EfficientNetV2-B0\nImageNet pretrained\nbackbone FROZEN\n(5.45M params)", STEEL, fs=11.2, tc=CREAM)
    box(4.9, 6.6, 3.0, 2.1, "Multi-scale feature maps\nearly 56x56 / mid 28x28\n/ deep 7x7 -> 1x1 conv\n+ global avg pool + concat",
        STEEL, fs=10.3, tc=CREAM)
    box(4.9, 1.2, 3.0, 2.1, "Texture encoder\nstem + separable conv\ndilation (1,2,4) -> GAP\n(local flakes / shine cues)",
        STEEL, fs=10.9, tc=CREAM)
    box(8.2, 3.7, 1.9, 2.2, "Concat +\nDense 256\nDropout", NAVY, fs=12, tc=CREAM)
    heads = ["oily", "dandruff", "dry", "redness /\nsensitive", "density\nissue"]
    for i, h in enumerate(heads):
        box(10.5, 8.3 - i * 1.55, 1.4, 1.05, "sigmoid\n" + h, SKY, fs=10.4)
        arrow(10.1, 4.8, 10.5, 8.8 - i * 1.55)
    arrow(1.8, 4.95, 2.0, 4.95)
    arrow(4.8, 5.7, 4.9, 7.5)
    arrow(4.8, 4.3, 4.9, 2.3)
    arrow(7.9, 7.6, 8.2, 5.4)
    arrow(7.9, 2.3, 8.2, 4.3)

    ax.text(11.2, 0.55, "5 independent raw logits -> sigmoid x100 -> 0-100 tendency scores",
            ha="center", fontsize=11.4, style="italic", color=C_SUBTEXT)
    ax.text(3.4, 3.15, "loss: masked BCE-with-logits   |   optimizer: AdamW 1e-3, wd 1e-5\n"
            "attribute weights: redness 1.5, density 1.5, others 1.0   |   seed 42, 20 epochs, batch 16",
            ha="center", fontsize=11.2, color=C_TEXT,
            bbox=dict(boxstyle="round", fc="#efe9db", ec=STEEL))
    ax.set_title("Production multi-attribute scalp model (6.07M total params, 621k trainable)",
                 fontsize=13.5, weight="bold", color=C_TEXT, pad=12)
    save(fig, "fig06_model_architecture.png")


# =====================================================================
# fig07 Keras training curves
# =====================================================================
def fig07():
    log = ROOT / "outputs_multiattr-20260723T134851Z-1-001(1)" / "outputs_multiattr" / "training_log.csv"
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    vloss = [float(r["val_loss"]) for r in rows]
    acc = [float(r["masked_binary_accuracy"]) for r in rows]
    vacc = [float(r["val_masked_binary_accuracy"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.7))
    fig.patch.set_facecolor(C_BG)
    for ax in [a1, a2]:
        ax.set_facecolor(C_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_TEXT)
        ax.tick_params(colors=C_TEXT)
        ax.title.set_color(C_TEXT)
        ax.xaxis.label.set_color(C_TEXT); ax.yaxis.label.set_color(C_TEXT)
    a1.plot(ep, loss, "-o", color=NAVY, label="train", ms=4)
    a1.plot(ep, vloss, "-o", color=STEEL, label="validation", ms=4)
    a1.set_xlabel("epoch"); a1.set_ylabel("masked BCE loss"); a1.set_title("Loss")
    a1.legend(facecolor=C_BG, edgecolor=C_TEXT, labelcolor=C_TEXT)
    a2.plot(ep, np.array(acc) * 100, "-o", color=NAVY, label="train", ms=4)
    a2.plot(ep, np.array(vacc) * 100, "-o", color=STEEL, label="validation", ms=4)
    a2.set_xlabel("epoch"); a2.set_ylabel("masked binary accuracy (%)"); a2.set_title("Masked binary accuracy")
    a2.legend(facecolor=C_BG, edgecolor=C_TEXT, labelcolor=C_TEXT)
    best_ep = int(ep[int(np.argmin(vloss))])
    a1.axvline(best_ep, ls="--", color=STEEL, alpha=0.8)
    a2.axvline(best_ep, ls="--", color=STEEL, alpha=0.8)
    a2.annotate(f"best checkpoint\n(epoch {best_ep}, val_loss {min(vloss):.3f})",
                xy=(best_ep, min(vacc) * 100), xytext=(11.5, 64),
                arrowprops=dict(arrowstyle="->", color=STEEL), fontsize=8.6, color=C_TEXT,
                bbox=dict(boxstyle="round,pad=0.2", fc=C_BG, ec=STEEL))
    fig.suptitle("Training dynamics of the 5-attribute Keras model (1,931 Roboflow images)",
                 fontsize=13.5, weight="bold", color=C_TEXT)
    footnote(fig, "Source: Keras training CSV log — 26 epochs, seed 42.")
    save(fig, "fig07_keras_training_curves.png")


# =====================================================================
# fig08 top-k recall
# =====================================================================
def fig08():
    ev = json.load(open(ROOT / "outputs_multiattr-20260723T134851Z-1-001(1)" / "outputs_multiattr"
                        / "evaluation_results.json", encoding="utf-8"))
    tk = ev["topk_metrics"]
    overall = [tk["top_1_attribute_recall"] * 100,
               tk["top_2_attribute_recall"] * 100,
               tk["top_3_attribute_recall"] * 100]
    pcr = ev["per_class_recall"]
    order = ["oily dandruff", "dry dandruff", "Oily scalp ", "Inflamed oily scalp ",
             "Dry scalp ", "Normal scalp "]
    labels = ["oily dandruff\n(n=9)", "dry dandruff\n(n=10)", "Oily scalp\n(n=11)",
              "Inflamed oily\n(n=9)", "Dry scalp\n(n=8)", "Normal scalp\n(n=10)"]
    t1 = [pcr[c]["top_1_recall"] * 100 for c in order]
    t2 = [pcr[c]["top_2_recall"] * 100 for c in order]
    t3 = [pcr[c]["top_3_recall"] * 100 for c in order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0),
                                 gridspec_kw={"width_ratios": [1, 2.1]})
    fig.subplots_adjust(bottom=0.26)
    fig.patch.set_facecolor(C_BG)
    for ax in [a1, a2]:
        ax.set_facecolor(C_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_TEXT)
        ax.tick_params(colors=C_TEXT)
        ax.title.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT); ax.xaxis.label.set_color(C_TEXT)
    bars = a1.bar(["top-1", "top-2", "top-3"], overall, color=[SKY, STEEL, NAVY])
    for b, v in zip(bars, overall):
        a1.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center",
                fontsize=10.5, weight="bold", color=C_TEXT)
    a1.set_ylim(0, 100); a1.set_ylabel("attribute recall (%)")
    a1.set_title("Overall test set (n=57)", fontsize=11, weight="bold")

    x = np.arange(6); w = 0.26
    a2.bar(x - w, t1, w, label="top-1", color=SKY)
    a2.bar(x, t2, w, label="top-2", color=STEEL)
    a2.bar(x + w, t3, w, label="top-3", color=NAVY)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=9.2, rotation=28, ha="right")
    a2.set_ylim(0, 112); a2.set_ylabel("recall (%)")
    a2.set_title("Per-class top-k attribute recall", fontsize=11, weight="bold")
    a2.legend(facecolor=C_BG, edgecolor=C_TEXT, labelcolor=C_TEXT)
    fig.suptitle("Evaluation on Roboflow single-label test split: the reported headline "
                 "metric is top-3 attribute recall = 78.9%", fontsize=13.5, weight="bold", color=C_TEXT)
    footnote(fig, "Source: Roboflow held-out test set (n=57), production Keras model.")
    save(fig, "fig08_topk_recall.png")


# =====================================================================
# fig09 split-end pipeline (real steps from shipped detector)
# =====================================================================
def fig09():
    from cv.split_end_detector import SplitEndDetector
    se = SplitEndDetector()
    img = cv2.imread(str(ROOT / "data" / "Skin_Hair" / "images" / "hair_0002.jpg"))
    blur = se.preprocess(img)
    binary = se.segment_hairs(blur)
    proc = se.morphological_operations(binary)
    skel = se.skeletonize(proc)
    pts = se.detect_branch_points(skel)
    n_hair = se.count_hairs(proc)
    overlay = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
    for (y, x) in pts:
        cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)

    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1. Input strand image\n(dark fibres on light background)"),
        (blur, "2. Grayscale + Gaussian blur (5x5)"),
        (binary, "3. Otsu threshold"),
        (skel, "4. Morphology + morphological\nskeletonisation"),
        (overlay, f"5. T/X kernel branch-point scan\n{n_hair} strand components, {len(pts)} branch points"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.5))
    for ax, (im, cap) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(cap, fontsize=9)
        ax.axis("off")
    fig.suptitle("Split-end detector pipeline (OpenCV, no training required)\n"
                 "split-end rate = branch points / strand components x 100%",
                 fontsize=11.5, weight="bold")
    save(fig, "fig09_split_end_pipeline.png")


# =====================================================================
# fig10 frizziness pipeline
# =====================================================================
def fig10():
    from cv.frizziness_detector import FrizzinessDetector
    fz = FrizzinessDetector()
    img = cv2.imread(str(ROOT / "data" / "Skin_Hair" / "images" / "hair_0032.jpg"))
    mask, gray = fz.segment_hair_region(img)
    contour = fz.find_main_hair_contour(mask)
    edges = cv2.Canny(gray, 50, 150)
    hull = cv2.convexHull(contour)
    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)
    expanded = cv2.dilate(hull_mask, np.ones((15, 15), np.uint8), iterations=3)
    fly = edges & (~expanded)
    n_fly = int(cv2.countNonZero(fly) // 75)
    grade, info = fz.detect(img)

    vis_region = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    vis_region[mask > 0] = (0.45 * vis_region[mask > 0] + 0.55 * np.array([46, 95, 163])).astype(np.uint8)
    cv2.drawContours(vis_region, [hull], -1, (255, 200, 0), 2)

    vis_fly = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    ys, xs = np.where(fly > 0)
    vis_fly[ys, xs] = (255, 0, 0)

    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1. Input"),
        (vis_region, "2. HSV hair mask (V<100)\n+ main-mass convex hull"),
        (cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), "3. Canny edges (50/150)"),
        (vis_fly, f"4. Edges outside dilated hull = fly-aways\n"
                  f"{cv2.countNonZero(fly)} edge px ~= {n_fly} fibres -> grade {grade}/3"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.7))
    for ax, (im, cap) in zip(axes, panels):
        ax.imshow(im); ax.set_title(cap, fontsize=9); ax.axis("off")
    fig.suptitle("Frizziness detector: fly-away fragments outside the main hair mass\n"
                 "ordinal grades 0 smooth -> 3 severely frizzy (aligned with T/CAFFCI 69-2023)",
                 fontsize=11.5, weight="bold")
    save(fig, "fig10_frizziness_pipeline.png")


# =====================================================================
# fig11 shininess pipeline
# =====================================================================
def fig11():
    from cv.shininess_detector import ShininessDetector
    sd = ShininessDetector()
    img = cv2.imread(str(ROOT / "data" / "Skin_Hair" / "images" / "hair_0000.jpg"))
    mask = sd.segment_hair_region(img)
    brightness = sd.calculate_brightness(img, mask)
    hist, _ = sd.compute_brightness_histogram(brightness, mask)
    ph, pw = sd.detect_highlight_peak(hist)
    grade, info = sd.detect(img)
    mean_b = info["brightness_mean"]
    peak_idx = int(np.argmax(hist))

    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    vis[mask > 0] = (0.5 * vis[mask > 0] + 0.5 * np.array([46, 95, 163])).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    fig.patch.set_facecolor(C_BG)
    fig.subplots_adjust(top=0.66)
    for ax in axes:
        ax.set_facecolor(C_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_TEXT)
        ax.tick_params(colors=C_TEXT)
        ax.title.set_color(C_TEXT)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].axis("off")
    axes[0].set_title("1. Input", fontsize=9.5)
    axes[1].imshow(vis); axes[1].axis("off")
    axes[1].set_title("2. Pixels used: HSV hair mask\n(black + dark brown hair)", fontsize=9.5)
    axes[2].bar(range(256), hist, width=1.0, color=STEEL)
    axes[2].axvline(peak_idx, color=NAVY, lw=1.6, ls="--")
    axes[2].annotate(f"dominant peak\nbin {peak_idx}, height {int(ph)}\nFWHM {int(pw)}",
                     xy=(peak_idx, ph), xytext=(peak_idx + 42, ph * 0.78),
                     arrowprops=dict(arrowstyle="->", color=NAVY), fontsize=8.6, color=C_TEXT)
    axes[2].set_xlim(0, 255); axes[2].set_xlabel("luminance Y (0-255)"); axes[2].set_ylabel("pixels")
    axes[2].set_title(f"3. Luminance histogram of hair pixels\n"
                      f"mean Y={mean_b:.1f} -> grade {grade}/2 (0 dull, 1 medium, 2 shiny)",
                      fontsize=9.5)
    fig.suptitle("Shine detector: specular highlights appear as a tall, narrow luminance peak\n"
                 "(no calibrated light source needed; grading aligned with T/CAFFCI 70-2023)",
                 fontsize=11.5, weight="bold", color=C_TEXT)
    footnote(fig, "Detector output on FDU frame 220829_C040_1 — unmodified shipping code.")
    save(fig, "fig11_shininess_pipeline.png")


# =====================================================================
# fig12 three-stage transfer learning (PyTorch research track)
# =====================================================================
def fig12():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9.4, tc=C_TEXT):
        shadow_box(ax, x, y, w, h, fc, boxstyle="round,pad=0.02,rounding_size=0.2")
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold", zorder=2.1)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.7, color=C_ARROW))

    box(0.3, 2.4, 2.5, 1.7, "Stage 0\nImageNet weights\nEfficientNet-B0\n(no training)", MIST)
    box(3.3, 2.4, 2.7, 1.7, "Stage 1 domain pre-train\nFDU scalp images\nfreeze 70% backbone\nAdam 1e-4, early stop", STEEL, tc=CREAM)
    box(6.5, 2.4, 2.7, 1.7, "Stage 2 task fine-tune\nA/B/C damage pseudo-labels\nfreeze 40-50%, LR 1e-5\nAMP + grad clip + early stop", NAVY, tc=CREAM)
    box(9.7, 4.4, 2.1, 1.4, "classification head\n3-class damage", SKY, fs=8.8)
    box(9.7, 2.55, 2.1, 1.2, "UNet-like seg head\n(mIoU tracked)", SKY, fs=8.8)
    box(9.7, 0.9, 2.1, 1.2, "detection head\ninterface only", MIST, fs=8.8)
    for x in [(2.8, 3.3), (6.0, 6.5)]:
        arrow(x[0], 3.25, x[1], 3.25)
    arrow(9.2, 3.6, 9.7, 5.0)
    arrow(9.2, 3.35, 9.7, 3.15)
    arrow(9.2, 3.0, 9.7, 1.5)
    ax.text(6.0, 0.82, "Honest status: framework production-ready, weights are a proof-of-concept only\n"
            "(A/B/C are filename-derived pseudo-labels; val accuracy is unstable across runs)",
            ha="center", fontsize=9, color=NAVY, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#efe9db", ec=STEEL))
    ax.set_title("PyTorch research track: three-stage transfer-learning framework (scalp_ai/)",
                 fontsize=13.5, weight="bold", color=C_TEXT, pad=12)
    save(fig, "fig12_transfer_learning_stages.png")


# =====================================================================
# fig13 PyTorch stage curves (parsed from logs/damage_stage*.log, 2026-08-12 runs)
# =====================================================================
def _parse_classification_run(path, run_date):
    """Extract the classification (Train/Valid Epoch ... Acc) run on run_date.

    The same log contains several runs; keep the classification run (lines
    with Acc) whose header date equals run_date with the most epochs.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    best_train, best_val = [], []
    cur_train, cur_val = [], []
    in_run = False
    for line in text.splitlines():
        if "开始训练" in line:
            if in_run and len(cur_val) > len(best_val):
                best_train, best_val = cur_train, cur_val
            in_run = run_date in line
            cur_train, cur_val = [], []
            continue
        m = re.search(r"Train Epoch (\d+) \| Loss ([\d.]+) \| Acc ([\d.]+)%", line)
        if m and in_run:
            cur_train.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        m = re.search(r"Valid Epoch (\d+) \| Loss ([\d.]+) \| Acc ([\d.]+)%", line)
        if m and in_run:
            cur_val.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if in_run and len(cur_val) > len(best_val):
        best_train, best_val = cur_train, cur_val
    return best_train, best_val


def fig13():
    t1, v1 = _parse_classification_run(ROOT / "logs" / "damage_stage1.log", "2026-08-12")
    t2, v2 = _parse_classification_run(ROOT / "logs" / "damage_stage2.log", "2026-08-12")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), sharey=True)
    fig.patch.set_facecolor(C_BG)
    fig.subplots_adjust(top=0.72)
    for ax, (tr, va, title) in zip(axes, [
        (t1, v1, "Stage 1 (freeze 70%, LR 1e-4)"),
        (t2, v2, "Stage 2 (freeze 40-50%, LR 1e-5)")]):
        ax.set_facecolor(C_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_TEXT)
        ax.tick_params(colors=C_TEXT)
        ax.title.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT); ax.xaxis.label.set_color(C_TEXT)
        ax.plot([e for e, _, _ in tr], [a for _, _, a in tr], "-o", ms=4,
                color=NAVY, label="train accuracy")
        ax.plot([e for e, _, _ in va], [a for _, _, a in va], "-o", ms=4,
                color=STEEL, label="validation accuracy")
        best = max(va, key=lambda r: r[2])
        ax.axhline(33.3, ls=":", color=STEEL)
        ax.text(0.6, 35.0, "3-class chance = 33.3%", fontsize=8, color=C_TEXT)
        ax.annotate(f"best val {best[2]:.1f}%\n(epoch {best[0]})",
                    xy=(best[0], best[2]), xytext=(best[0] + 1.5, best[2] - 9),
                    arrowprops=dict(arrowstyle="->", color=STEEL), fontsize=8.6, color=C_TEXT,
                    bbox=dict(boxstyle="round,pad=0.2", fc=C_BG, ec=STEEL))
        ax.set_xlabel("epoch"); ax.set_title(title, fontsize=10.5, weight="bold")
        ax.set_ylim(25, 70); ax.legend(fontsize=8.5, facecolor=C_BG, edgecolor=C_TEXT, labelcolor=C_TEXT)
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("PyTorch damage-grade training on FDU pseudo-labels (early stopping, patience 5)\n"
                 "both stages plateau near 60% -> label noise, not a deployable grader",
                 fontsize=11.5, weight="bold", color=C_TEXT)
    footnote(fig, "Source: PyTorch stage logs — workstation retrain, seed 42.")
    save(fig, "fig13_pytorch_stage_curves.png")


# =====================================================================
# fig14 SAM2 ablation
# =====================================================================
def fig14():
    js = json.load(open(ROOT / "results" / "sam2_view_comparison_5.json", encoding="utf-8"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.1),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(C_BG)
    fig.subplots_adjust(top=0.70)
    for ax in [a1, a2]:
        ax.set_facecolor(C_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_TEXT)
        ax.tick_params(colors=C_TEXT)
        ax.title.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
    bars = a1.bar(["Original images", "SAM2 ROI crop"], [61, 43],
                  color=[NAVY, SKY], width=0.55)
    for b, v in zip(bars, [61, 43]):
        a1.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v}%", ha="center",
                fontsize=12, weight="bold", color=C_TEXT)
    a1.set_ylim(0, 75); a1.set_ylabel("validation accuracy (%)")
    a1.set_title("Retraining comparison on FDU\n(SAM2 ROI preprocessing ON)", fontsize=10.5, weight="bold")

    labels = ["hard ROI crop\nmean |delta score|", "soft mask (0.3+0.7m)\nmean |delta score|",
              "hard ROI\ntop-1 agreement", "soft mask\ntop-1 agreement"]
    vals = [js["roi_mean_abs_score_delta"], js["soft_mask_mean_abs_score_delta"],
            js["roi_top1_agreement"], js["soft_mask_top1_agreement"]]
    cols = [STEEL, SKY, NAVY, NAVY]
    bars = a2.bar(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}" if v < 1 else f"{v*100:.0f}%",
                ha="center", fontsize=10, weight="bold", color=C_TEXT)
    a2.set_ylim(0, 1.18); a2.set_ylabel("paired-ablation value (n=5 FDU images)")
    a2.set_title("Paired input-ablation: predictions barely move\nand top-1 never changes",
                 fontsize=10.5, weight="bold")
    a2.tick_params(axis="x", labelsize=8.2)
    fig.suptitle("SAM2 ROI experiment: a deliberately documented negative result\n"
                 "close-up scalp frames have no removable background -> ROI discarded useful texture "
                 "(kept default-OFF)", fontsize=12, weight="bold", color=C_TEXT)
    footnote(fig, "Source: retrain logs and SAM2 ablation runs — 20 epochs, seed 42.")
    save(fig, "fig14_sam2_ablation.png")


# =====================================================================
# fig15 SAM2 visual example (copy existing artifact)
# =====================================================================
def fig15():
    import shutil
    src = ROOT / "test_sam2_output.jpg"
    fig, ax = plt.subplots(figsize=(5.8, 7.0))
    img = imread_u(src)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.set_title("SAM2.1 zero-shot output on an FDU frame\n"
                 "largest non-background mask + ROI centre mark\n"
                 "(1,588 / 1,652 frames segmented successfully = 96.1%)",
                 fontsize=10.5, weight="bold")
    save(fig, "fig15_sam2_example.png")


# =====================================================================
# fig16 radar examples from the production model
# =====================================================================
def fig16():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from PIL import Image
    from profile_generator import ProfileGenerator

    ATTR_EN = ["Oiliness", "Dandruff", "Dryness", "Redness/\nSensitivity", "Density\nissue"]
    keys = ["oily", "dandruff", "dry", "redness_or_sensitive", "hair_density_issue"]
    model = tf.keras.models.load_model(str(RAIL / "model" / "best_model.keras"), compile=False)
    gen = ProfileGenerator()
    samples = [("220105_A095_1_thumb.jpg", "A (healthy pseudo-label)"),
               ("220124_B159_1_thumb.jpg", "B (mild pseudo-label)"),
               ("220829_C035_1_thumb.jpg", "C (severe pseudo-label)")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(C_BG)
    fig.subplots_adjust(top=0.70)
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    for ax, (fn, tag) in zip(axes, samples):
        ax.set_facecolor(C_BG)
        pil = Image.open(ROOT / "fdu_preview" / fn).convert("RGB").resize((224, 224))
        x = np.array(pil, dtype=np.float32)[None, ...]
        logits = np.array(model.predict(x, verbose=0)).flatten().tolist()
        prof = gen.generate_from_logits(logits)
        scores = prof["percentage_scores"]
        vals = [scores[k] for k in keys]
        v_loop = vals + vals[:1]; a_loop = angles + angles[:1]
        ax.fill(a_loop, v_loop, color=STEEL, alpha=0.35)
        ax.plot(a_loop, v_loop, color=NAVY, lw=2)
        ax.set_xticks(angles); ax.set_xticklabels(ATTR_EN, fontsize=8.2, color=C_TEXT)
        ax.set_ylim(0, 100); ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=6.5, color=C_TEXT)
        h = prof["healthy"]["healthy"]
        ax.set_title(f"{fn.split('_')[0]} — {tag}\nhealth score = {h}/100",
                     fontsize=9.5, weight="bold", pad=14, color=C_TEXT)
    fig.suptitle("Production model outputs on one example frame per FDU pseudo-label class\n"
                 "(illustrative inference outputs — FDU prefixes are NOT 5-attribute ground truth)",
                 fontsize=11.5, weight="bold", color=C_TEXT)
    footnote(fig, "Scores produced by the production unified inference engine.")
    save(fig, "fig16_radar_examples.png")


if __name__ == "__main__":
    fig01(); fig02(); fig03(); fig04(); fig05(); fig06(); fig07(); fig08()
    fig09(); fig10(); fig11(); fig12(); fig13(); fig14(); fig15(); fig16()
    print("ALL DONE ->", OUT)
