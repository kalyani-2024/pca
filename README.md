# PCA — Practical Guide + Gene Expression Application

A self-contained, code-first package to learn PCA and see it applied to a real high-dimensional biological dataset.

## Contents

| File | Purpose |
|---|---|
| [PCA_Complete_Guide.md](PCA_Complete_Guide.md) | Practical guide with 10 embedded diagrams + 1 animated GIF. Covers workflow, choosing k, failure modes, variants, comparisons, a short Q&A, and a cheat-sheet. |
| [Gene_Expression_PCA.ipynb](Gene_Expression_PCA.ipynb) | Hands-on notebook — loads a real 801×20 531 cancer RNA-Seq dataset and runs the full PCA workflow with code + explanations in each cell. |
| [assets/](assets/) | Figures used by the guide. Regenerable via the script below. |
| [scripts/generate_visuals.py](scripts/generate_visuals.py) | Regenerates every figure from scratch (`python scripts/generate_visuals.py`). |
| [requirements.txt](requirements.txt) | Pip dependencies. |

### Visuals in the guide

| # | File | What it shows |
|---|---|---|
| 1 | `01_intuition_ellipse.png` | PCA as rotation onto variance-aligned axes |
| 2 | `02_variance_sweep.gif` | Animated sweep of axis angle vs projected variance |
| 3 | `03_projection_reconstruction.png` | Perpendicular residuals = reconstruction error |
| 4 | `04_scree_and_cumulative.png` | Scree plot + cumulative explained variance |
| 5 | `05_standardization_effect.png` | Why PCA needs standardized features |
| 6 | `06_outlier_sensitivity.png` | One outlier rotates PC1 |
| 7 | `07_kernel_pca.png` | Linear PCA fails on rings; RBF kernel PCA unfolds them |
| 8 | `08_pca_vs_lda.png` | Unsupervised variance axis vs supervised axis |
| 9 | `09_svd_diagram.png` | `X = U Σ Vᵀ` block diagram |
| 10 | `10_reconstruction_k.png` | Digit reconstruction quality vs k |

---

## Dataset

**Gene Expression Cancer RNA-Seq** — UCI Machine Learning Repository (ID 401), a curated subset of **The Cancer Genome Atlas (TCGA) Pan-Cancer project** (Weinstein et al., *Nature Genetics* 2013).

- **Samples:** 801 patient tumor samples
- **Features:** 20 531 gene expression values (log-transformed RSEM RNA-Seq)
- **Classes:** 5 cancer types — BRCA (breast), KIRC (kidney), COAD (colon), LUAD (lung), PRAD (prostate)
- **URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz`
- **License:** UCI ML Repository — free for research and teaching.

The notebook downloads and caches the archive on first run.

---

## Installation & Running

### 1. Open this folder
```bash
cd path/to/pca
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Regenerate the visuals
```bash
python scripts/generate_visuals.py
```
Produces everything in `assets/` from scratch (~10 seconds). Only needed if you want to tweak the figures.

### 5. Launch the notebook
```bash
jupyter lab            # or:  jupyter notebook
```
Open `Gene_Expression_PCA.ipynb` and run cells top-to-bottom. First execution downloads ~50 MB once into `./data/`; subsequent runs use the cache.

### 6. Optional — run the whole notebook headless
```bash
jupyter nbconvert --to notebook --execute Gene_Expression_PCA.ipynb \
                  --output executed.ipynb
```

### System requirements
- Python 3.9+
- ~2 GB RAM during PCA fit
- Internet access on the first run (dataset download)

---

## Suggested Study Order

1. Read [PCA_Complete_Guide.md](PCA_Complete_Guide.md) end-to-end.
2. Open [Gene_Expression_PCA.ipynb](Gene_Expression_PCA.ipynb) and run each cell; read the markdown before the code executes.
3. Change one thing at a time in the notebook (k, scaler, solver) and re-run — the fastest way to build intuition.
4. Revisit the Q&A and cheat-sheet at the end of the guide.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| SSL error downloading dataset | Update Python and `certifi`; or manually download the tarball from the URL above into `./data/`. |
| `ImportError: sklearn` | `pip install --upgrade scikit-learn` (>= 1.1). |
| Kernel dies during PCA fit | Ensure `svd_solver="randomized"` (default in notebook); avoid `"full"` on this dataset. |
| 3-D plot doesn't render | Ensure `matplotlib >= 3.4`. |

---

## License & Attribution

- Code: MIT-style — free to adapt.
- Dataset citation: Weinstein, J. N., Collisson, E. A., Mills, G. B., Shaw, K. R. M., Ozenberger, B. A., Ellrott, K., Shmulevich, I., Sander, C., & Stuart, J. M. (2013). The Cancer Genome Atlas Pan-Cancer analysis project. *Nature Genetics*, 45(10), 1113-1120.
