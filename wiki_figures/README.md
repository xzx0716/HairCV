# Wiki Figures

`generate_figures.py` regenerates every chart on the iGEM wiki Detection
page (`fig01`-`fig16`) from real project artifacts — training CSV logs,
evaluation JSON, the shipped detector code and the production Keras model.
No number in the figures is invented; each is read from project files or
produced by running the shipped code.

## Regenerate one figure

```bash
python generate_figures.py            # all figures
python -c "import generate_figures as g; g.fig06()"   # a single figure
```

Outputs PNG files (cream `#f7f3ea` background, navy/steel-blue palette,
150 dpi) into `../images/`. Before publishing on the wiki, PNGs are
converted to AVIF (quality 50-55, max width 1400 px) to stay inside the
GitLab Pages bundle budget.
