"""
发质CV分析主程序

支持命令行和Python API调用
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict

from preprocessing import ImagePreprocessor
from cuticle_detector import CuticleDetector
from split_end_detector import SplitEndDetector
from frizziness_detector import FrizzinessDetector
from shininess_detector import ShininessDetector
from scorer import ScoreMapper


class HairCVSystem:
    """发质CV分析系统"""

    def __init__(self):
        """初始化系统"""
        self.preprocessor = ImagePreprocessor()
        self.cuticle_detector = CuticleDetector()
        self.split_end_detector = SplitEndDetector()
        self.frizziness_detector = FrizzinessDetector()
        self.shininess_detector = ShininessDetector()
        self.scorer = ScoreMapper()

    def analyze_image(self, image_path: str, image_type: str) -> Dict:
        """
        分析单张图像

        Args:
            image_path: 图像路径
            image_type: 图像类型 ("microscope", "split_end", "macro")

        Returns:
            分析结果字典
        """
        # 加载图像
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        # 预处理
        if image_type == "microscope":
            processed = self.preprocessor.crop_microscope_roi(image)
            processed = self.preprocessor.resize_image(processed)
            grade, details = self.cuticle_detector.detect(processed)
            score_result = self.scorer.map_cuticle_grade(grade)
            return {
                "indicator": "毛鳞片完整度",
                "raw_value": grade,
                "raw_description": ["完整", "轻微磨损", "中度磨损", "严重磨损"][grade],
                "score": score_result.score,
                "confidence": score_result.confidence,
                "details": details
            }

        elif image_type == "split_end":
            processed = self.preprocessor.crop_split_end_roi(image)
            rate, details = self.split_end_detector.detect(processed)
            score_result = self.scorer.map_split_end_rate(rate)
            return {
                "indicator": "分叉率",
                "raw_value": rate,
                "score": score_result.score,
                "confidence": score_result.confidence,
                "details": details
            }

        elif image_type == "macro":
            processed = self.preprocessor.crop_macro_roi(image)
            grade, details = self.frizziness_detector.detect(processed)
            score_result = self.scorer.map_frizziness_grade(grade)
            return {
                "indicator": "毛躁度",
                "raw_value": grade,
                "raw_description": details["grade_description"],
                "score": score_result.score,
                "confidence": score_result.confidence,
                "details": details
            }

        elif image_type == "shininess":
            processed = self.preprocessor.crop_macro_roi(image)
            grade, details = self.shininess_detector.detect(processed)
            score_result = self.scorer.map_shininess_grade(grade)
            return {
                "indicator": "光泽度",
                "raw_value": grade,
                "raw_description": details["grade_description"],
                "score": score_result.score,
                "confidence": score_result.confidence,
                "details": details
            }

        else:
            raise ValueError(f"未知的图像类型: {image_type}")

    def analyze_all(self,
                    microscope_path: Optional[str] = None,
                    split_end_path: Optional[str] = None,
                    macro_path: Optional[str] = None,
                    shininess_path: Optional[str] = None) -> Dict:
        """
        综合分析

        Args:
            microscope_path: 显微镜图像路径
            split_end_path: 微距图像路径
            macro_path: 宏观图像路径
            shininess_path: 光泽度图像路径

        Returns:
            综合分析结果
        """
        results = {}

        if microscope_path:
            results["毛鳞片完整度"] = self.analyze_image(microscope_path, "microscope")

        if split_end_path:
            results["分叉率"] = self.analyze_image(split_end_path, "split_end")

        if macro_path:
            results["毛躁度"] = self.analyze_image(macro_path, "macro")

        if shininess_path:
            results["光泽度"] = self.analyze_image(shininess_path, "shininess")

        return results


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="发质CV分析系统")
    parser.add_argument("--microscope", "-m", help="显微镜图像路径")
    parser.add_argument("--split-end", "-s", help="微距图像路径")
    parser.add_argument("--macro", "-f", help="宏观图像路径")
    parser.add_argument("--shininess", "-n", help="光泽度图像路径")
    parser.add_argument("--output", "-o", help="输出JSON文件路径")
    parser.add_argument("--demo", "-d", action="store_true", help="启动Gradio Demo")

    args = parser.parse_args()

    if args.demo:
        # 启动Demo
        from demo import main as demo_main
        demo_main()
        return

    # 分析图像
    system = HairCVSystem()
    results = system.analyze_all(
        microscope_path=args.microscope,
        split_end_path=args.split_end,
        macro_path=args.macro,
        shininess_path=args.shininess
    )

    # 输出结果
    output = json.dumps(results, ensure_ascii=False, indent=2)
    print(output)

    # 保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\n结果已保存到: {args.output}")

    # 输出AHP格式
    print("\n=== AHP综合评分 ===")
    for indicator, result in results.items():
        print(f"{indicator}: {result['score']}分")


if __name__ == "__main__":
    main()
