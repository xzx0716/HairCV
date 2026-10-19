"""
CV结果评分映射模块

将CV算法输出的原始结果转换为谢振兴AHP模型需要的0-10分单项数值
"""

import json
from typing import Dict, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ScoreResult:
    """评分结果"""
    indicator: str  # 指标名称
    raw_value: Any  # 原始值
    score: float  # 0-10分
    confidence: float = 1.0  # 置信度


class ScoreMapper:
    """评分映射器"""

    # 毛鳞片完整度评分映射
    CUTICLE_GRADE_TO_SCORE = {
        0: 10,  # 完整 = 10分
        1: 7,   # 轻微磨损 = 7分
        2: 4,   # 中度磨损 = 4分
        3: 1,   # 严重磨损 = 1分
    }

    # 分叉率评分映射（基于分叉率百分比）
    SPLIT_END_RATE_TO_SCORE = [
        (5, 10),    # <5% = 10分
        (15, 7),    # 5%-15% = 7分
        (30, 4),    # 15%-30% = 4分
        (100, 1),   # >30% = 1分
    ]

    # 毛躁度评分映射
    FRIZZINESS_GRADE_TO_SCORE = {
        0: 10,  # 无毛躁 = 10分
        1: 7,   # 轻微毛躁 = 7分
        2: 4,   # 中度毛躁 = 4分
        3: 1,   # 严重毛躁 = 1分
    }

    # 光泽度评分映射
    SHININESS_GRADE_TO_SCORE = {
        0: 2,   # 暗淡 = 2分
        1: 6,   # 中等光泽 = 6分
        2: 10,  # 有光泽 = 10分
    }

    def __init__(self):
        """初始化评分映射器"""
        pass

    def map_cuticle_grade(self, grade: int) -> ScoreResult:
        """
        映射毛鳞片等级到评分

        Args:
            grade: 毛鳞片等级 (0-3)

        Returns:
            ScoreResult
        """
        score = self.CUTICLE_GRADE_TO_SCORE.get(grade, 0)
        return ScoreResult(
            indicator="毛鳞片完整度",
            raw_value=grade,
            score=float(score),
            confidence=0.85  # CV精度
        )

    def map_split_end_rate(self, rate: float) -> ScoreResult:
        """
        映射分叉率到评分

        Args:
            rate: 分叉率 (%)

        Returns:
            ScoreResult
        """
        score = 0
        for threshold, s in self.SPLIT_END_RATE_TO_SCORE:
            if rate < threshold:
                score = s
                break

        # 计算置信度（基于与阈值的距离）
        confidence = 1.0
        for threshold, s in self.SPLIT_END_RATE_TO_SCORE:
            if rate < threshold:
                # 离阈值越近，置信度越低
                if rate > threshold * 0.8:
                    confidence = 0.9
                break

        return ScoreResult(
            indicator="分叉率",
            raw_value=f"{rate:.2f}%",
            score=float(score),
            confidence=min(confidence, 0.9)  # CV精度
        )

    def map_frizziness_grade(self, grade: int) -> ScoreResult:
        """
        映射毛躁度等级到评分

        Args:
            grade: 毛躁度等级 (0-3)

        Returns:
            ScoreResult
        """
        score = self.FRIZZINESS_GRADE_TO_SCORE.get(grade, 0)
        return ScoreResult(
            indicator="毛躁度",
            raw_value=grade,
            score=float(score),
            confidence=0.80  # CV精度
        )

    def map_shininess_grade(self, grade: int) -> ScoreResult:
        """
        映射光泽度等级到评分

        Args:
            grade: 光泽度等级 (0-2)

        Returns:
            ScoreResult
        """
        score = self.SHININESS_GRADE_TO_SCORE.get(grade, 0)
        return ScoreResult(
            indicator="光泽度",
            raw_value=grade,
            score=float(score),
            confidence=0.70  # CV精度
        )

    def map_all(self,
                cuticle_grade: int = None,
                split_end_rate: float = None,
                frizziness_grade: int = None,
                shininess_grade: int = None) -> Dict[str, ScoreResult]:
        """
        映射所有指标

        Args:
            cuticle_grade: 毛鳞片等级
            split_end_rate: 分叉率
            frizziness_grade: 毛躁度等级
            shininess_grade: 光泽度等级

        Returns:
            所有评分结果的字典
        """
        results = {}

        if cuticle_grade is not None:
            results["cuticle"] = self.map_cuticle_grade(cuticle_grade)

        if split_end_rate is not None:
            results["split_end"] = self.map_split_end_rate(split_end_rate)

        if frizziness_grade is not None:
            results["frizziness"] = self.map_frizziness_grade(frizziness_grade)

        if shininess_grade is not None:
            results["shininess"] = self.map_shininess_grade(shininess_grade)

        return results

    def to_json(self, results: Dict[str, ScoreResult]) -> str:
        """
        将评分结果转换为JSON格式

        Args:
            results: 评分结果字典

        Returns:
            JSON字符串
        """
        output = {}
        for key, result in results.items():
            output[key] = {
                "indicator": result.indicator,
                "raw_value": result.raw_value,
                "score": result.score,
                "confidence": result.confidence
            }
        return json.dumps(output, ensure_ascii=False, indent=2)

    def to_ahp_format(self, results: Dict[str, ScoreResult]) -> Dict[str, float]:
        """
        转换为谢振兴AHP模型需要的格式

        Args:
            results: 评分结果字典

        Returns:
            AHP格式的字典（指标名:分数）
        """
        ahp_output = {}
        for result in results.values():
            ahp_output[result.indicator] = result.score
        return ahp_output


def create_scorer() -> ScoreMapper:
    """
    创建评分映射器

    Returns:
        ScoreMapper实例
    """
    return ScoreMapper()


# 评分标准说明文档
SCORE_STANDARDS = """
# CV结果量化标准 v1.0

## 一、评分映射规则

### 1.1 毛鳞片完整度
| 等级 | 原始值 | 评分 | 说明 |
|------|--------|------|------|
| 0级 | 完整 | 10分 | 毛鳞片完整，边缘整齐 |
| 1级 | 轻微磨损 | 7分 | 边缘轻微模糊<10% |
| 2级 | 中度磨损 | 4分 | 边缘模糊10%-30% |
| 3级 | 严重磨损 | 1分 | 边缘模糊>30% |

### 1.2 分叉率
| 分叉率范围 | 评分 | 说明 |
|------------|------|------|
| <5% | 10分 | 基本无分叉 |
| 5%-15% | 7分 | 轻度分叉 |
| 15%-30% | 4分 | 中度分叉 |
| >30% | 1分 | 严重分叉 |

### 1.3 毛躁度
| 等级 | 原始值 | 评分 | 说明 |
|------|--------|------|------|
| 0级 | 无毛躁 | 10分 | 无明显飞丝 |
| 1级 | 轻微毛躁 | 7分 | 飞丝<10根 |
| 2级 | 中度毛躁 | 4分 | 飞丝10-30根 |
| 3级 | 严重毛躁 | 1分 | 飞丝>30根 |

### 1.4 光泽度
| 等级 | 原始值 | 评分 | 说明 |
|------|--------|------|------|
| 0级 | 暗淡 | 2分 | 无明显光泽 |
| 1级 | 中等光泽 | 6分 | 有一定光泽感 |
| 2级 | 有光泽 | 10分 | 光泽感强 |

## 二、输出格式

### JSON格式
```json
{
  "cuticle": {
    "indicator": "毛鳞片完整度",
    "raw_value": 1,
    "score": 7.0,
    "confidence": 0.85
  },
  "split_end": {
    "indicator": "分叉率",
    "raw_value": "12.5%",
    "score": 7.0,
    "confidence": 0.9
  },
  "frizziness": {
    "indicator": "毛躁度",
    "raw_value": 2,
    "score": 4.0,
    "confidence": 0.8
  },
  "shininess": {
    "indicator": "光泽度",
    "raw_value": 1,
    "score": 6.0,
    "confidence": 0.7
  }
}
```

### AHP格式（仅分数）
```json
{
  "毛鳞片完整度": 7.0,
  "分叉率": 7.0,
  "毛躁度": 4.0,
  "光泽度": 6.0
}
```

## 三、置信度说明

CV结果的置信度反映了算法对预测结果的自信程度：

| 指标 | 目标置信度 | 说明 |
|------|-----------|------|
| 毛鳞片完整度 | ≥85% | 4分类任务 |
| 分叉率 | ≤±10%误差 | 计数任务 |
| 毛躁度 | ≥80% | 4分类任务 |
| 光泽度 | ≥70% | 3分类任务 |

**注意：** 所有CV结果需与人工标注结果做一致性校验，最终以人工复核为准。
"""
