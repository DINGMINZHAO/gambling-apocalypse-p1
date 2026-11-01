# Gambling Apocalypse Part I — LaTeX Report

This project includes the complete English research report, **eight verified empirical figures**, four result tables, and the source files needed to rebuild charts that were redrawn from CSV. The manuscript uses A4 single-column journal-style typesetting with Times-like fonts and numbered equations, tables, and figures.

## Folder layout

```text
main.tex                    # Master LaTeX document
sections/                   # Approved English report, one section per file
figures/                    # All 8 real report figures, ready to compile
figure_data/                # Original research PNGs and verified CSV sources
generate_figures.py         # Recreate the 8 figures (optional)
FIGURE_SOURCES.md           # Provenance for every chart
build/main.pdf              # Precompiled PDF included in the ZIP
```

## Work in VS Code on macOS

1. Unzip the project under your repository, for example `~/Desktop/gambling-p1/reports/gambling-apocalypse-latex/`.
2. Open the folder in VS Code. The **LaTeX Workshop** extension and a TeX distribution (MacTeX or BasicTeX with `newtx`, `booktabs`, `microtype`, etc.) are recommended.
3. Open `main.tex` and use LaTeX Workshop to build, or run:

```bash
mkdir -p build
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
open build/main.pdf
```

The images are already in `figures/`. You do not have to run Python or download any source data to compile.

If you want to regenerate the report figures from the saved original images and CSV files, run:

```bash
python3 -m pip install pandas numpy matplotlib pillow
python3 generate_figures.py
latexmk -g -pdf -interaction=nonstopmode -outdir=build main.tex
```

The chart generator does **not** process raw trade data or fit new models. It only redraws figures from existing research results.

## Editorial notes

- `Abstract` is unnumbered. Accordingly, `Data and Research Design` is Section 1 in the PDF.
- There is no separate Introduction, as agreed.
- The title is exactly **Gambling Apocalypse Part I: Who Is Actually Smart Money?**
- Source manuscript, original chart sources, and result figures are provided for reproducibility.
- This is the **illustrated LaTeX draft**, ready for further visual editing in VS Code. No unverified or synthetic research numbers have been introduced.
