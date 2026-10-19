from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math


ATTRIBUTE_NAMES = [
    "oily",
    "dandruff",
    "dry",
    "redness_or_sensitive",
    "hair_density_issue",
]


ATTRIBUTE_DISPLAY_NAMES = {
    "oily": "油性倾向",
    "dandruff": "头屑倾向",
    "dry": "干燥倾向",
    "redness_or_sensitive": "敏感红斑倾向",
    "hair_density_issue": "毛发密度异常倾向",
}


ATTRIBUTE_SHORT_NAMES = {
    "oily": "油性",
    "dandruff": "头屑",
    "dry": "干燥",
    "redness_or_sensitive": "敏感红斑",
    "hair_density_issue": "毛发密度异常",
}


@dataclass(frozen=True)
class AttributeProfile:
    raw_probabilities: Dict[str, float]
    percentage_scores: Dict[str, int]
    detected_attributes: Dict[str, int]
    possible_attributes: Dict[str, int]
    healthy: Dict[str, int]
    summary: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw_probabilities": self.raw_probabilities,
            "percentage_scores": self.percentage_scores,
            "detected_attributes": self.detected_attributes,
            "possible_attributes": self.possible_attributes,
            "healthy": self.healthy,
            "summary": self.summary,
        }


class ProfileGenerator:
    """Generate front-end friendly scalp attribute profiles.

    Model logits are converted into sigmoid scores, then reported as visual
    tendency scores. These are not medical diagnosis probabilities.
    """

    def __init__(
        self,
        attribute_names: Sequence[str] = ATTRIBUTE_NAMES,
        detected_threshold: int = 60,
        possible_threshold: int = 30,
    ):
        if not 0 <= possible_threshold <= 100:
            raise ValueError("possible_threshold must be between 0 and 100.")
        if not 0 <= detected_threshold <= 100:
            raise ValueError("detected_threshold must be between 0 and 100.")
        if possible_threshold > detected_threshold:
            raise ValueError("possible_threshold should not exceed detected_threshold.")

        self.attribute_names = list(attribute_names)
        self.detected_threshold = int(detected_threshold)
        self.possible_threshold = int(possible_threshold)

    @staticmethod
    def sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def probabilities_from_logits(
        self, logits: Sequence[float]
    ) -> Dict[str, float]:
        if len(logits) != len(self.attribute_names):
            raise ValueError(
                f"Expected {len(self.attribute_names)} logits, got {len(logits)}."
            )
        return {
            name: float(self.sigmoid(float(logit)))
            for name, logit in zip(self.attribute_names, logits)
        }

    def scores_from_probabilities(
        self, probabilities: Mapping[str, float]
    ) -> Dict[str, int]:
        scores = {}
        for name in self.attribute_names:
            probability = float(probabilities[name])
            score = int(round(max(0.0, min(1.0, probability)) * 100))
            scores[name] = score
        return scores

    def generate_from_logits(self, logits: Sequence[float]) -> Dict[str, object]:
        probabilities = self.probabilities_from_logits(logits)
        return self.generate_from_probabilities(probabilities)

    def generate_from_probabilities(
        self, probabilities: Mapping[str, float]
    ) -> Dict[str, object]:
        scores = self.scores_from_probabilities(probabilities)
        return self.generate_from_scores(scores, probabilities=probabilities)

    def generate_from_scores(
        self,
        scores: Mapping[str, int],
        probabilities: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, object]:
        percentage_scores = {
            name: int(max(0, min(100, round(float(scores[name])))))
            for name in self.attribute_names
        }
        if probabilities is None:
            raw_probabilities = {
                name: percentage_scores[name] / 100.0 for name in self.attribute_names
            }
        else:
            raw_probabilities = {
                name: float(probabilities[name]) for name in self.attribute_names
            }

        detected = {
            name: score
            for name, score in percentage_scores.items()
            if score >= self.detected_threshold
        }
        possible = {
            name: score
            for name, score in percentage_scores.items()
            if self.possible_threshold <= score < self.detected_threshold
        }
        symptom_max = max(percentage_scores.values()) if percentage_scores else 0
        healthy_score = 100 - symptom_max

        profile = AttributeProfile(
            raw_probabilities=raw_probabilities,
            percentage_scores=percentage_scores,
            detected_attributes=detected,
            possible_attributes=possible,
            healthy={"healthy": healthy_score},
            summary=self.build_summary(percentage_scores, detected, possible),
        )
        return profile.to_dict()

    def build_summary(
        self,
        scores: Mapping[str, int],
        detected: Mapping[str, int],
        possible: Mapping[str, int],
    ) -> str:
        if not detected and not possible:
            return "未见明显异常头皮特征，整体更接近健康状态。"

        detected_parts = []
        for name, score in sorted(
            detected.items(), key=lambda item: item[1], reverse=True
        ):
            detected_parts.append(
                f"{ATTRIBUTE_DISPLAY_NAMES.get(name, name)}评分为 {score}/100，{self.level_text(score)}"
            )

        possible_parts = []
        for name, score in sorted(
            possible.items(), key=lambda item: item[1], reverse=True
        ):
            possible_parts.append(
                f"{ATTRIBUTE_SHORT_NAMES.get(name, name)}倾向（{score}/100）"
            )

        absent_names = [
            ATTRIBUTE_SHORT_NAMES.get(name, name)
            for name in self.attribute_names
            if scores[name] < self.possible_threshold
        ]

        sentences = []
        if detected_parts:
            sentences.append("您的头皮" + "；".join(detected_parts) + "。")
        if possible_parts:
            prefix = "您的头皮存在" if not sentences else "同时存在"
            sentences.append(prefix + "一定" + "、".join(possible_parts) + "。")
        if absent_names:
            sentences.append("暂未见明显" + "、".join(absent_names) + "特征。")

        return "".join(sentences)

    @staticmethod
    def level_text(score: int) -> str:
        if score < 30:
            return "未见明显特征"
        if score < 60:
            return "存在轻度倾向"
        if score < 80:
            return "特征较明显"
        return "特征非常明显"


def demo_profiles() -> List[Dict[str, object]]:
    generator = ProfileGenerator()
    examples = [
        {
            "oily": 56,
            "dandruff": 43,
            "dry": 12,
            "redness_or_sensitive": 18,
            "hair_density_issue": 21,
        },
        {
            "oily": 72,
            "dandruff": 64,
            "dry": 8,
            "redness_or_sensitive": 35,
            "hair_density_issue": 22,
        },
        {
            "oily": 12,
            "dandruff": 18,
            "dry": 24,
            "redness_or_sensitive": 10,
            "hair_density_issue": 15,
        },
    ]
    return [generator.generate_from_scores(scores) for scores in examples]


if __name__ == "__main__":
    for profile in demo_profiles():
        print(profile)
