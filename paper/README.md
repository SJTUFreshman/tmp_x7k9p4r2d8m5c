# Paper source

**Reallocating Reasoning across Models via Learned Collaboration**

This directory contains the manuscript source accompanying the reproduction artifacts. The main file is [`iclr2027_conference.tex`](iclr2027_conference.tex). It describes Collaborative Learning, including model slicing and pyramid supervision, and evaluates five benchmarks across four task categories.

## Source inventory

| File or directory | Purpose |
| --- | --- |
| `iclr2027_conference.tex` | Main manuscript |
| `main_results.tex` | Table 1, including all baseline and CL results |
| `ablation_results.tex` | Figure 3 inclusion and caption |
| `appendix.tex` | Protocol, training, baseline, metric, and case-study details |
| `math_commands.tex` | Shared mathematical notation |
| `iclr2027_conference.bib` | Bibliography database |
| `iclr2027_conference.bst` | Bibliography style |
| `iclr2027_conference.sty`, `natbib.sty`, `fancyhdr.sty` | Bundled typesetting dependencies |
| `figures/Figure1.pdf` | Concept: reallocating reasoning through learned collaboration |
| `figures/Figure2.pdf` | Method overview: model slicing and pyramid supervision |
| `figures/Figure3.pdf` | Ablation results |
| `figures/Figure4.pdf` | Computation cost versus task performance |
| `figures/Figure5.pdf` | Post-training interaction topology |
| `figures/logo_*.svg` | Four trajectory-type icons used in the Figure 5 caption |

All local `\input`, bibliography, PDF-figure, and SVG-figure references in the manuscript are included. Author-internal notes, source archives, and unrelated upload assets are not part of this source directory. The main results are a TeX table, not an additional figure file.

## Build

Use a recent TeX distribution with `latexmk`, pdfLaTeX, BibTeX, and the packages named in the manuscript preamble. The `svg` package also requires Inkscape to convert the caption icons; the command below enables that conversion.

From the repository root:

```bash
cd paper
latexmk -pdf -shell-escape -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

The expected output is `iclr2027_conference.pdf`. Local source dependencies have been checked; successful compilation additionally depends on the installed TeX packages and Inkscape. The manuscript uses the ICLR conference style, `algorithm`/`algpseudocode`, `nicematrix`, `svg`, and standard table, graphics, and cross-reference packages.

## Source status

The TeX files preserve the anonymous manuscript source. The author block contains conference-template placeholders and is not authoritative author or citation metadata. No acceptance or publication status is implied by the conference template.

For experiment artifacts, the reported table, and the MATH scoring and evaluation entry point, see the [repository README](../README.md).
