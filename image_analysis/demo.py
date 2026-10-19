"""
发质CV分析Demo

基于Gradio的图形界面Demo，展示CV模块的功能
"""

import gradio as gr
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple

from preprocessing import ImagePreprocessor
from cuticle_detector import CuticleDetector
from split_end_detector import SplitEndDetector
from frizziness_detector import FrizzinessDetector
from shininess_detector import ShininessDetector
from scorer import ScoreMapper


class HairCVAnalyzer:
    """发质CV分析器"""

    def __init__(self):
        """初始化分析器"""
        self.preprocessor = ImagePreprocessor()
        self.cuticle_detector = CuticleDetector()
        self.split_end_detector = SplitEndDetector()
        self.frizziness_detector = FrizzinessDetector()
        self.shininess_detector = ShininessDetector()
        self.scorer = ScoreMapper()

    def analyze_microscope(self, image: np.ndarray) -> Tuple[str, str, str]:
        """
        分析显微镜图（毛鳞片完整度）

        Args:
            image: 显微镜图像

        Returns:
            (等级描述, 评分结果, 详细信息)
        """
        # 预处理
        processed = self.preprocessor.crop_microscope_roi(image)
        processed = self.preprocessor.resize_image(processed)

        # 检测
        grade, details = self.cuticle_detector.detect(processed)

        # 评分
        score_result = self.scorer.map_cuticle_grade(grade)

        # 格式化输出
        grade_desc = {
            0: "毛鳞片完整（0级）",
            1: "毛鳞片轻微磨损（1级）",
            2: "毛鳞片中度磨损（2级）",
            3: "毛鳞片严重磨损（3级）"
        }.get(grade, "未知")

        score_text = f"评分: {score_result.score}分"
        details_text = f"边缘密度: {details['edge_density']:.4f}"

        return grade_desc, score_text, details_text

    def analyze_split_end(self, image: np.ndarray) -> Tuple[str, str, str]:
        """
        分析微距图（分叉率）

        Args:
            image: 微距图像

        Returns:
            (分叉率描述, 评分结果, 详细信息)
        """
        # 预处理
        processed = self.preprocessor.crop_split_end_roi(image)

        # 检测
        rate, details = self.split_end_detector.detect(processed)

        # 评分
        score_result = self.scorer.map_split_end_rate(rate)

        # 格式化输出
        rate_desc = f"分叉率: {rate:.2f}%"
        score_text = f"评分: {score_result.score}分"
        details_text = f"分叉点数: {details['split_end_count']}, 发丝数: {details['total_hairs']}"

        return rate_desc, score_text, details_text

    def analyze_macro_frizziness(self, image: np.ndarray) -> Tuple[str, str, str]:
        """
        分析宏观图（毛躁度）

        Args:
            image: 宏观图像

        Returns:
            (毛躁度描述, 评分结果, 详细信息)
        """
        # 预处理
        processed = self.preprocessor.crop_macro_roi(image)

        # 检测
        grade, details = self.frizziness_detector.detect(processed)

        # 评分
        score_result = self.scorer.map_frizziness_grade(grade)

        # 格式化输出
        grade_desc = details['grade_description']
        score_text = f"评分: {score_result.score}分"
        details_text = f"飞丝数量: {details['flyaway_count']}"

        return grade_desc, score_text, details_text

    def analyze_macro_shininess(self, image: np.ndarray) -> Tuple[str, str, str]:
        """
        分析宏观光泽度图

        Args:
            image: 宏观光泽度专用图像

        Returns:
            (光泽度描述, 评分结果, 详细信息)
        """
        # 预处理
        processed = self.preprocessor.crop_macro_roi(image)

        # 检测
        grade, details = self.shininess_detector.detect(processed)

        # 评分
        score_result = self.scorer.map_shininess_grade(grade)

        # 格式化输出
        grade_desc = details['grade_description']
        score_text = f"评分: {score_result.score}分"
        details_text = f"平均亮度: {details['brightness_mean']:.2f}"

        return grade_desc, score_text, details_text

    def analyze_all(self,
                   microscope_image: np.ndarray,
                   split_end_image: np.ndarray,
                   macro_image: np.ndarray,
                   shininess_image: np.ndarray) -> str:
        """
        综合分析所有图像

        Args:
            microscope_image: 显微镜图像
            split_end_image: 微距分叉率图像
            macro_image: 宏观毛躁度图像
            shininess_image: 宏观光泽度图像

        Returns:
            JSON格式的综合结果
        """
        results = {}

        if microscope_image is not None:
            grade, _ = self.cuticle_detector.detect(
                self.preprocessor.resize_image(
                    self.preprocessor.crop_microscope_roi(microscope_image)
                )
            )
            results["毛鳞片完整度"] = {
                "raw_value": grade,
                "score": self.scorer.map_cuticle_grade(grade).score
            }

        if split_end_image is not None:
            rate, _ = self.split_end_detector.detect(
                self.preprocessor.crop_split_end_roi(split_end_image)
            )
            results["分叉率"] = {
                "raw_value": f"{rate:.2f}%",
                "score": self.scorer.map_split_end_rate(rate).score
            }

        if macro_image is not None:
            grade, _ = self.frizziness_detector.detect(
                self.preprocessor.crop_macro_roi(macro_image)
            )
            results["毛躁度"] = {
                "raw_value": grade,
                "score": self.scorer.map_frizziness_grade(grade).score
            }

        if shininess_image is not None:
            grade, _ = self.shininess_detector.detect(
                self.preprocessor.crop_macro_roi(shininess_image)
            )
            results["光泽度"] = {
                "raw_value": grade,
                "score": self.scorer.map_shininess_grade(grade).score
            }

        return json.dumps(results, ensure_ascii=False, indent=2)


def create_demo():
    """创建Gradio Demo"""

    analyzer = HairCVAnalyzer()

    with gr.Blocks(title="发质CV分析Demo") as demo:
        gr.Markdown("# 发质CV分析Demo")
        gr.Markdown("上传发质图像，自动分析毛鳞片完整度、分叉率、毛躁度、光泽度")

        with gr.Tab("单项分析"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 毛鳞片完整度（显微镜图）")
                    microscope_input = gr.Image(type="numpy", label="显微镜图像")
                    microscope_btn = gr.Button("分析毛鳞片")
                    microscope_output1 = gr.Textbox(label="等级")
                    microscope_output2 = gr.Textbox(label="评分")
                    microscope_output3 = gr.Textbox(label="详细信息")

                with gr.Column():
                    gr.Markdown("### 分叉率（微距图）")
                    split_end_input = gr.Image(type="numpy", label="微距图像")
                    split_end_btn = gr.Button("分析分叉率")
                    split_end_output1 = gr.Textbox(label="分叉率")
                    split_end_output2 = gr.Textbox(label="评分")
                    split_end_output3 = gr.Textbox(label="详细信息")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 毛躁度（宏观图）")
                    frizziness_input = gr.Image(type="numpy", label="宏观图像")
                    frizziness_btn = gr.Button("分析毛躁度")
                    frizziness_output1 = gr.Textbox(label="毛躁度")
                    frizziness_output2 = gr.Textbox(label="评分")
                    frizziness_output3 = gr.Textbox(label="详细信息")

                with gr.Column():
                    gr.Markdown("### 光泽度（光泽度专用图）")
                    shininess_input = gr.Image(type="numpy", label="光泽度图像")
                    shininess_btn = gr.Button("分析光泽度")
                    shininess_output1 = gr.Textbox(label="光泽度")
                    shininess_output2 = gr.Textbox(label="评分")
                    shininess_output3 = gr.Textbox(label="详细信息")

        with gr.Tab("综合分析"):
            gr.Markdown("### 上传所有图像进行综合分析")
            all_microscope = gr.Image(type="numpy", label="显微镜图像")
            all_split_end = gr.Image(type="numpy", label="微距图像")
            all_macro = gr.Image(type="numpy", label="宏观图像")
            all_shininess = gr.Image(type="numpy", label="光泽度图像")
            all_btn = gr.Button("综合分析")
            all_output = gr.JSON(label="综合结果")

        # 绑定事件
        microscope_btn.click(
            analyzer.analyze_microscope,
            inputs=[microscope_input],
            outputs=[microscope_output1, microscope_output2, microscope_output3]
        )

        split_end_btn.click(
            analyzer.analyze_split_end,
            inputs=[split_end_input],
            outputs=[split_end_output1, split_end_output2, split_end_output3]
        )

        frizziness_btn.click(
            analyzer.analyze_macro_frizziness,
            inputs=[frizziness_input],
            outputs=[frizziness_output1, frizziness_output2, frizziness_output3]
        )

        shininess_btn.click(
            analyzer.analyze_macro_shininess,
            inputs=[shininess_input],
            outputs=[shininess_output1, shininess_output2, shininess_output3]
        )

        all_btn.click(
            analyzer.analyze_all,
            inputs=[all_microscope, all_split_end, all_macro, all_shininess],
            outputs=[all_output]
        )

    return demo


def main():
    """主函数"""
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
