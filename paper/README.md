# Paper source

LaTeX source of the AABA4ET 2026 workshop paper, set up for the camera-ready version
(`\usepackage[dblblindworkshop,final]{neurips_2026}`).

`paper.pdf` in this folder is the anonymous version that was submitted for review. Compiling
`paper.tex` overwrites it with the camera-ready build.

## Build

```bash
cd paper
pdflatex paper.tex && pdflatex paper.tex
```

Or upload this folder to Overleaf. Everything it needs is here: the NeurIPS 2026 style files
and `figures/`. The figures are generated from `results/` by `make figures` in the repo root.

## Still to do for camera-ready

Search `paper.tex` for `\TODO`. Each one shows up in red in the PDF:

- an affiliation and email for each author
- the acknowledgements and funding disclosure

Keep the body (up to the acknowledgements) within 4 pages. References and the appendix do not
count.

Note: the bundled `environ.sty` breaks the `ack` environment in `final` mode, so the
acknowledgements use `\acksection` directly. The output is the same.
