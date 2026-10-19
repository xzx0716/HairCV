# -*- coding: utf-8 -*-
"""
头皮/发质视觉分析统一推理引擎

整合项目内全部视觉模型成果：
  1. 主力：Keras 头皮多属性模型（colab 训练成果，best_model.keras）
     → 5 维度倾向分：油性/头屑/干燥/敏感红斑/毛发密度 + 健康分
  2. 传统CV发丝指标（src/）：分叉率 / 毛躁度 / 光泽度
  3. 可选：PyTorch 发质损伤模型（scalp_ai，EfficientNet-B0）
  4. 可选：SAM2 头皮区域ROI裁剪（手机自拍有背景时使用）

注意：所有模型输出为视觉倾向评分，仅供参考，不构成医疗诊断。
"""

import os
import sys
import warnings
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------- 路径配置 ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERAS_MODEL_PATH = os.path.join(
    ROOT,
    "outputs_multiattr-20260723T134851Z-1-001(1)",
    "outputs_multiattr",
    "best_model.keras",
)
COLAB_DIR = os.path.join(ROOT, "colab(1)", "colab")
IMG_SIZE = 224

# 让 profile_generator 可导入
if COLAB_DIR not in sys.path:
    sys.path.insert(0, COLAB_DIR)
# 让 src / scalp_ai 可导入
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# =============================================================
# 1. 主力模型：Keras 头皮多属性模型
# =============================================================
class ScalpAttributeModel:
    """头皮5属性多标签模型（EfficientNetV2B0，sigmoid多头输出）"""

    ATTRIBUTE_CN = {
        "oily": "油性倾向",
        "dandruff": "头屑倾向",
        "dry": "干燥倾向",
        "redness_or_sensitive": "敏感红斑倾向",
        "hair_density_issue": "毛发密度异常",
    }

    def __init__(self, model_path: str = KERAS_MODEL_PATH):
        import tensorflow as tf
        self.tf = tf
        print(f"[引擎] 加载头皮多属性模型: {os.path.basename(model_path)}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        from profile_generator import ProfileGenerator
        self.generator = ProfileGenerator()
        print("[引擎] 头皮多属性模型就绪")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Args:
            image: PIL RGB 图像
        Returns:
            dict: percentage_scores / healthy / summary / detected / possible
        """
        img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)  # 模型内部自带 Rescaling(1/255)
        x = np.expand_dims(arr, 0)
        logits = self.model.predict(x, verbose=0)
        logits = np.array(logits).flatten().tolist()
        profile = self.generator.generate_from_logits(logits)
        return profile


# =============================================================
# 2. 传统CV发丝指标（无需训练，开箱即用）
# =============================================================
class TraditionalHairCV:
    """分叉率 / 毛躁度 / 光泽度 传统CV检测"""

    def __init__(self):
        from src.split_end_detector import SplitEndDetector
        from src.frizziness_detector import FrizzinessDetector
        from src.shininess_detector import ShininessDetector
        self.split_detector = SplitEndDetector()
        self.frizz_detector = FrizzinessDetector()
        self.shine_detector = ShininessDetector()
        print("[引擎] 传统CV发丝检测器就绪")

    @staticmethod
    def _pil_to_bgr(image: Image.Image) -> np.ndarray:
        import cv2
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        import cv2
        bgr = self._pil_to_bgr(image)
        results = {}

        # Split-end rate
        try:
            rate, info = self.split_detector.detect(bgr)
            results["split_end"] = {
                "name": "Split-end rate",
                "value": round(float(rate), 1),
                "unit": "%",
                "split_count": int(info["split_end_count"]),
                "hair_count": int(info["total_hairs"]),
                "detail": f"{info['split_end_count']} split points / {info['total_hairs']} hairs",
            }
        except Exception as e:
            results["split_end"] = {"name": "Split-end rate", "error": str(e)[:60]}

        # Frizziness
        try:
            grade, info = self.frizz_detector.detect(bgr)
            grade_names = {0: "Smooth", 1: "Slightly frizzy", 2: "Moderately frizzy", 3: "Severely frizzy"}
            results["frizziness"] = {
                "name": "Frizziness",
                "grade": int(grade),
                "level": grade_names.get(int(grade), "Unknown"),
                "flyaway_count": int(info.get("flyaway_count", -1)) if isinstance(info, dict) else -1,
            }
        except Exception as e:
            results["frizziness"] = {"name": "Frizziness", "error": str(e)[:60]}

        # Shine
        try:
            grade, info = self.shine_detector.detect(bgr)
            grade_names = {0: "Dull", 1: "Medium shine", 2: "Shiny"}
            results["shininess"] = {
                "name": "Shine",
                "grade": int(grade),
                "level": grade_names.get(int(grade), "Unknown"),
            }
        except Exception as e:
            results["shininess"] = {"name": "Shine", "error": str(e)[:60]}

        return results


# =============================================================
# 3. 可选：PyTorch 发质损伤模型
# =============================================================
class TorchDamageModel:
    """Hair damage 3-class model (Healthy / Mild / Severe)"""

    LEVELS = {0: "Healthy", 1: "Mild damage", 2: "Severe damage"}

    def __init__(self):
        import torch
        self.torch = torch
        from scalp_ai.configs import DAMAGE_CFG, CHECKPOINT_DIR
        from scalp_ai.models import build_damage_model
        from scalp_ai.datasets import build_classification_transforms
        import albumentations as A

        ckpt = os.path.join(CHECKPOINT_DIR, "damage_stage2_final.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(CHECKPOINT_DIR, "damage_stage2_best.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError("未找到发质模型权重")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_damage_model(
            backbone_name=DAMAGE_CFG.backbone_name, pretrained=False,
            num_seg_classes=DAMAGE_CFG.num_seg_classes,
            num_classes=DAMAGE_CFG.num_classes_stage2,
        ).to(self.device)
        state = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()
        self.transform = build_classification_transforms(DAMAGE_CFG.aug, is_train=False)
        print(f"[引擎] PyTorch发质模型就绪 ({self.device})")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        import cv2
        rgb = np.array(image.convert("RGB"))
        transformed = self.transform(image=rgb)
        x = transformed["image"].unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            out = self.model(x, stage="stage2")
            probs = self.torch.softmax(out["logits"], dim=1).squeeze().cpu().numpy()
        idx = int(probs.argmax())
        return {
            "name": "Hair damage assessment",
            "level": self.LEVELS[idx],
            "confidence": round(float(probs[idx]) * 100, 1),
            "distribution": {self.LEVELS[i]: round(float(p) * 100, 1)
                             for i, p in enumerate(probs)},
        }


# =============================================================
# 4. 可选：SAM2 ROI 裁剪
# =============================================================
def sam2_crop_scalp(image: Image.Image) -> Optional[Image.Image]:
    """用SAM2零样本分割并裁剪最大区域（手机自拍去背景）。失败返回None。"""
    try:
        import cv2
        from scalp_ai.preprocessing.sam2_roi import SAM2HairSegmentor
        seg = SAM2HairSegmentor()  # 默认 sam2.1_b
        rgb = np.array(image.convert("RGB"))
        _, mask = seg.segment_image(rgb)
        if mask is None:
            return None
        roi = seg.extract_roi(rgb, mask)
        return Image.fromarray(roi)
    except Exception as e:
        print(f"[引擎] SAM2裁剪跳过: {str(e)[:80]}")
        return None


# =============================================================
# 统一分析入口
# =============================================================
class UnifiedScalpEngine:
    """统一视觉分析引擎：一次加载，多次分析"""

    def __init__(self, enable_torch: bool = False, enable_sam2: bool = False):
        self.scalp_model = ScalpAttributeModel()
        self.trad_cv = TraditionalHairCV()
        self.torch_model = None
        if enable_torch:
            try:
                self.torch_model = TorchDamageModel()
            except Exception as e:
                print(f"[引擎] PyTorch发质模型不可用: {str(e)[:80]}")
        self._sam2_ready = False
        if enable_sam2:
            try:
                import cv2  # noqa
                from scalp_ai.preprocessing.sam2_roi import SAM2HairSegmentor  # noqa
                self._sam2_ready = True
            except Exception:
                self._sam2_ready = False

    def analyze(self, image: Image.Image, use_sam2: bool = False) -> Dict[str, Any]:
        """
        对一张图片运行全部视觉分析。

        Returns:
            dict:
              scalp   - 头皮5属性评分（主力）
              hair_cv - 分叉率/毛躁度/光泽度（传统CV）
              damage  - 发质损伤3分类（可选）
              note    - 使用提示
        """
        if image is None:
            return {"error": "No image received."}

        # 可选SAM2预处理（手机自拍去背景）
        processed = image
        sam2_used = False
        if use_sam2 and self._sam2_ready:
            cropped = sam2_crop_scalp(image)
            if cropped is not None:
                processed = cropped
                sam2_used = True

        result: Dict[str, Any] = {}

        # 1. 头皮多属性（主力模型）
        result["scalp"] = self.scalp_model.predict(processed)

        # 2. 传统CV发丝指标
        result["hair_cv"] = self.trad_cv.analyze(processed)

        # 3. PyTorch发质损伤（可选）
        if self.torch_model is not None:
            try:
                result["damage"] = self.torch_model.predict(processed)
            except Exception as e:
                result["damage"] = {"error": str(e)[:80]}

        result["note"] = (
            "SAM2 scalp-region crop applied." if sam2_used else
            "Full image analysed. For selfies with a busy background, enable 'Smart ROI crop'."
        )
        result["disclaimer"] = ("Results are AI visual tendency analysis for reference only; "
                                "not a medical diagnosis.")
        return result


# 便捷单图函数（脚本测试用）
def analyze_image(path: str) -> Dict[str, Any]:
    engine = UnifiedScalpEngine()
    img = Image.open(path)
    return engine.analyze(img)


if __name__ == "__main__":
    import json
    test_img = os.path.join(ROOT, "data", "发质修复", "FDU", "Images", "220105_A095_1.jpg")
    eng = UnifiedScalpEngine()
    img = Image.open(test_img)
    r = eng.analyze(img)
    print("\n=== 头皮5属性 ===")
    for k, v in r["scalp"]["percentage_scores"].items():
        print(f"  {ScalpAttributeModel.ATTRIBUTE_CN.get(k, k)}: {v}/100")
    print(f"  健康分: {r['scalp']['healthy']['healthy']}/100")
    print(f"  总结: {r['scalp']['summary']}")
    print("\n=== 传统CV ===")
    print(json.dumps(r["hair_cv"], ensure_ascii=False, indent=2))
