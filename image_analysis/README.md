# Image Analysis

Reusable CV library for the hair-quality readout. Every module is plain
OpenCV/NumPy — no deep-learning runtime required.

## Modules

| File | Purpose |
| --- | --- |
| `preprocessing.py` | `ImagePreprocessor` (resize/normalise/augment, 224x224 default), `LabelMeParser` |
| `cuticle_detector.py` | `CuticleDetector` — edge/texture features for cuticle integrity grading |
| `split_end_detector.py` | `SplitEndDetector` — Otsu threshold + skeleton branching for split-end rate |
| `frizziness_detector.py` | `FrizzinessDetector` — flyaway-strand counting, 0-3 grade |
| `shininess_detector.py` | `ShininessDetector` — brightness/highlight statistics, 0-2 grade |
| `scorer.py` | `ScoreMapper` — maps detector grades/rates to 0-10 scores, JSON and AHP-format output |
| `main.py` | `HairCVSystem` — CLI and Python API entry point |
| `demo.py` | Stand-alone Gradio demo of the library |

## Input assumption

The fibre detectors expect **dark strands on a light background** (hair
strand photographs). Their outputs on scalp-surface macro frames are not
interpretable.

## Usage

```python
import sys
sys.path.insert(0, "hair_benchmark/image_analysis")

from main import HairCVSystem

system = HairCVSystem()
result = system.analyze("strand_photo.jpg")   # returns JSON-ready dict
```

Unit tests live in `../tests/test_haircv.py` (20 tests, all passing).
