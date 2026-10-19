# Tests

Pytest suite for the `../image_analysis` library.

```bash
pip install pytest numpy opencv-python-headless
pytest hair_benchmark/tests/test_haircv.py -v
```

Covers preprocessing (resize, min-max normalisation, augmentation), the
four fibre detectors' grading rules, and the 0-10 score mapping with JSON
and AHP-format output. 20 tests, all passing.
