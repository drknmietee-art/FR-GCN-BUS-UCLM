# FR-GCN: Fuzzy-Rough Graph Convolutional Network for BUS-UCLM

Uncertainty-aware breast ultrasound lesion segmentation on the **BUS-UCLM** dataset.
The model represents each ultrasound image as a superpixel graph. Fuzzy c-means gives a soft
lesion membership per node. A fuzzy-rough approximation gives a boundary uncertainty per node.
A graph convolutional network then classifies each node, and the labels are mapped back to pixels.

## Repository contents

| File | Description |
|------|-------------|
| `FR_GCN_BUS_UCLM.m` | Complete MATLAB R2024b pipeline (data, features, GCN training, evaluation). Also caches a final model to `results/model.mat`. |
| `FR_GCN_figures.m` | Generates all manuscript figures (1 to 6) as MATLAB `.fig`, PNG (300 dpi) and PDF into `results/figures`. |
| `run_baselines.m` | Runs FCM+threshold, SLIC+SVM, plain GCN and FR-GCN under one 5-fold split; writes `comparison.csv`, `stats.csv`, `ablation.csv`. |
| `FR_GCN_BUS_UCLM_Paper.docx` | Full manuscript with native Word equations, tables, and figure placeholders. |
| `build_paper.js` | Node script that regenerates the `.docx` (uses `docx-js`). |
| `README.md` | This file. |

## Dataset

BUS-UCLM (Vallez et al., 2025), 683 breast ultrasound images from 38 patients.
Masks are RGB coded: **green = benign (174)**, **red = malignant (90)**, **black = normal / background (419)**.

- Paper: https://www.nature.com/articles/s41597-025-04562-3
- Data: https://data.mendeley.com/datasets/7fvgj4jsp7/1

Place the dataset so the folder layout is:

```
BUS-UCLM Breast ultrasound lesion segmentation dataset/
  BUS-UCLM/
    images/   ALWI_000.png ...
    masks/    ALWI_000.png ...
```

Set `CFG.dataRoot` in `FR_GCN_BUS_UCLM.m` if your path differs.

## Requirements

MATLAB R2024b with:

- Image Processing Toolbox
- Deep Learning Toolbox
- Fuzzy Logic Toolbox (optional; an internal fuzzy c-means fallback is included)
- Statistics and Machine Learning Toolbox (optional)

## How to run

```matlab
>> FR_GCN_BUS_UCLM
```

For a fast smoke test, set `CFG.maxImages = 40;` near the top of the script.
Outputs (metrics table, box plot, raw metrics) are written to `./results/`.

### Generating the figures

After training, run:

```matlab
>> FR_GCN_figures
```

This reuses `results/model.mat` when present, otherwise it trains a compact
model and caches it. It writes the following to `results/figures/` as PNG
(300 dpi) and vector PDF:

- `Figure1_pipeline` schematic block diagram
- `Figure2_membership_boundary` fuzzy membership and fuzzy-rough boundary maps
- `Figure3_graph` superpixels and the region adjacency graph
- `Figure4_qualitative` input, ground truth, and FR-GCN prediction
- `Figure5_boxplot` per-image Dice and Jaccard distribution
- `Figure6_convergence` training loss against epoch

## Method summary

1. Preprocess: grayscale, anisotropic diffusion for speckle, intensity normalisation.
2. Superpixels (SLIC) become graph nodes; shared borders become edges.
3. Fuzzy c-means membership and Lukasiewicz fuzzy-rough lower / upper approximation.
4. Node features: intensity, GLCM texture, fuzzy membership, rough boundary uncertainty.
5. Two-layer GCN with symmetric normalised adjacency and self-loops.
6. Uncertainty-weighted cross entropy plus soft Dice loss.
7. Node probabilities mapped to pixels; metrics reported over five-fold cross-validation.

## Reproducing the manuscript

```bash
npm install docx
node build_paper.js      # writes FR_GCN_BUS_UCLM_Paper.docx
```

## Citation

If you use this code, please cite the dataset and this repository.

```
Vallez, N., Bueno, G., Deniz, O., Rienda, M. A., & Pastor, C. (2025).
BUS-UCLM: Breast ultrasound lesion segmentation dataset. Scientific Data, 12, 242.
```

## License

MIT
