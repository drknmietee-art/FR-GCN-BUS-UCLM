# FR-GCN — MATLAB implementation

Reference MATLAB implementation for *Fuzzy-Rough Edge Gating for Graph-Based
Breast Ultrasound Lesion Segmentation*. It reproduces every table, figure and
statistic in the manuscript.

## Quick start

```matlab
cd matlab
FRGCN_Main          % runs all six stages end to end
```

or stage by stage:

```matlab
FRGCN_Extract       % 1. images -> superpixel graphs (cache_graphs.mat)
FRGCN_RunAll        % 2. the single seeded run: all models, 5 folds x 5 seeds
FRGCN_Dense         % 3. U-Net / nnU-Net baselines   (Deep Learning Toolbox)
FRGCN_Uncertainty   % 4. B(p) vs entropy vs MC dropout
FRGCN_Diagnostics   % 5. oracle ceiling, descriptor ceiling, edge weights
FRGCN_Tables        % 6a. every manuscript table -> results/*.csv
FRGCN_Figures       % 6b. every figure -> results/figures/*.png
```

Expected layout:

```
<parent>/
    BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM/
        images/*.png
        masks/*.png
    matlab/            <- this folder; run MATLAB from here
```

For a first end-to-end check on a reduced budget (about 20x faster, **not** the
published protocol):

```matlab
FRGCN_QUICK = true; FRGCN_RunAll
```

## Figures

Figure numbering matches the paper, and each figure is callable on its own.

| Figure | Content | Needs | Produced by |
|---|---|---|---|
| 1 | Pipeline schematic | nothing | `fr_figpipeline` |
| 2 | Superpixels + region adjacency graph | `cache_graphs.mat` | `fr_figsuperpixels` |
| 3 | Per-image Dice distribution | `perimage_all.csv` | `fr_figdice` |
| 4 | Component ablation | `T8_ablation.csv` | `fr_figablation` |
| 5 | Uncertainty quality | `U2_`/`U3_*.csv` | `fr_figuncertainty` |
| 6 | Training / validation curve | `cache_graphs.mat` | `fr_figtraining` |
| 7 | Decision-threshold sweep | `U5_threshold.csv` | `fr_figthreshold` |

Numbering matches the manuscript: Figure 6 is the convergence curve in
Section 8.8 ("Training fit"), Figure 7 is the threshold sweep a few paragraphs
later ("Operating point and post-processing").

**After training, this is the order that fills everything in:**

```matlab
FRGCN_Uncertainty     % writes U2_, U3_, U5_  ->  Figures 5 and 6
FRGCN_Tables          % writes T8_ablation    ->  Figure 4
FRGCN_Figures         % renders all seven
```

Figure 4 needs the ablation *table*, so `FRGCN_Tables` must run before it;
Figures 5 and 6 need the uncertainty study. `FRGCN_Figures` skips any figure
whose input is missing and names the script that produces it, so the rest
still render. To redo just one:

```matlab
fr_figablation        % or fr_figuncertainty, fr_figthreshold, ...
```

Figure 6 is the one figure that costs time: the main experiment keeps only the
best checkpoint, so this retrains fold 1 for the full epoch budget to record
both curves. It also writes `results/training_curve.csv`.

Outputs land in `results/figures/` as `Figure1_...png` through
`Figure7_...png` at 200 dpi.

## Toolbox requirements

The pipeline is deliberately close to dependency-free. SLIC, the GLCM, the
graph network, the bootstrap, the Wilcoxon test, the ROC area, the box plots
and the CSV I/O are all implemented here rather than called from a toolbox.

| Stage | Requires |
|---|---|
| `FRGCN_Extract`, `FRGCN_RunAll`, `FRGCN_Uncertainty`, `FRGCN_Diagnostics`, `FRGCN_Tables` | base MATLAB only (`imread`, `imresize`, `rgb2gray`) |
| `fr_hd95`, `fr_largestcc` | `bwdist`, `bwlabel` (Image Processing Toolbox) |
| `FRGCN_Dense` | Deep Learning Toolbox (`unetLayers`, `trainNetwork`) |
| `FRGCN_Figures` | base MATLAB graphics |

Skipping `FRGCN_Dense` still produces every other table; the dense rows are
simply absent from Table 6.

## What is where

**Core numerics**

```
fr_fcm            Fuzzy c-means, Eqs (1)-(3)
fr_fuzzyrough     Lukasiewicz lower/upper approximations and B(p), Eqs (4)-(6)
fr_normadj        Symmetric normalised adjacency, Eq. (8); spectrum in (-1,1]
fr_gcn_init/forward/backward/step   L-layer GCN with analytic gradients + Adam
fr_loss           Uncertainty-weighted cross-entropy + soft Dice, Eqs (11)-(13)
fr_train          Training with validation-based early stopping
fr_metrics        Overlap metrics, Eqs (15)-(19)
fr_hd95           95th-percentile Hausdorff distance, Eq. (20)
fr_signrank / fr_ttest_rel / fr_holm / fr_bca / fr_auroc / fr_ece   statistics
fr_patientfolds   Patient-grouped folds by greedy bin packing
fr_slic           SLIC superpixels, implemented here (no toolbox needed)
fr_maskedglcm     Region-masked co-occurrence statistics
fr_features_v2    The fourteen additional node descriptors
fr_seed/rand/randn/randperm   Deterministic xorshift32 stream
```

**Why the RNG is home-grown.** `RandStream` behaviour varies across releases
and does not exist in Octave, so the paper's results would not be exactly
reproducible. The xorshift32 stream in `fr_seed`/`fr_rand` makes every run
bit-reproducible on any MATLAB or Octave version.

**Why two kernel scales.** `cfg.sigmaE` gates the graph edges and must
*separate* adjacent regions, so it wants to be small. `cfg.sigmaB` enters the
fuzzy-rough approximations and needs a neighbourhood wide enough for the lower
and upper bounds to differ. With a single shared scale, `B(p)` collapses to
zero in a 20-dimensional descriptor space and the descriptor is silently
crippled — which is exactly what happened in the first version of this work.

**The lesion-cluster rule matters.** `fr_fuzzyrough` picks the lesion cluster
by the **darker centroid**, an unsupervised prior. An earlier version selected
it by correlating the fuzzy memberships against the ground-truth node labels;
because the graph builder runs on test images too, that leaked held-out labels
into the node features. The leak is within-image, so patient-level grouping
does not protect against it.

## Verification

This implementation was executed and cross-checked before delivery.

**Numerically cross-checked against the Python reference on identical inputs**
(agreement to 8 decimal places unless noted):

- fuzzy-rough lower/upper approximation and `B(p)`
- region-masked GLCM contrast, homogeneity, energy, correlation
- Dice, Jaccard, sensitivity, specificity, precision
- HD95
- paired *t*-test; Holm adjustment; weighted ROC area and average precision
- Wilcoxon signed-rank (exact agreement for n ≳ 50; the paper's tests use
  n = 264. SciPy switches to the exact null distribution below n ≈ 50, so
  small-sample values can differ in the third decimal)
- BCa bootstrap (agrees to ~0.001; the residual is bootstrap Monte-Carlo
  noise from a different resampling stream, not a difference in method)
- fuzzy edge-weight distribution on adjacent node pairs: MATLAB mean 0.319
  (p10 0.001, p90 0.815) against Python 0.322 (0.004, 0.791)

**Gradient check.** `fr_gcn_backward` agrees with central finite differences to
a maximum relative error of 5e-8 for 2, 3 and 4 layers. The spectral radius of
the normalised operator is 1.000000, as Lemma 1 requires.

**End-to-end run.** `FRGCN_Extract`, `FRGCN_RunAll`, `FRGCN_Uncertainty`,
`FRGCN_Tables` and `FRGCN_Diagnostics` were executed under GNU Octave 8.4 on
16 real BUS-UCLM images from 6 patients. All stages completed and every
manuscript table was produced. On that small subset the qualitative findings of
the paper already reproduce: FR-GCN-lean > no-propagation control > FR-GCN >>
plain GCN; removing the fuzzy edge weights is by far the most damaging
ablation; the no-propagation control beats the plain GCN significantly
(p = 0.003); and `B(p)` ranks error near chance while predictive entropy and
MC dropout do not.

**Figures executed and inspected.** All seven figures were rendered under
Octave with the gnuplot toolkit and checked visually. Two bar charts were
corrected in the process: `bar()`/`barh()` in a loop does not honour its width
argument consistently across MATLAB and Octave, which silently merged adjacent
bars into one band, so both now draw explicit rectangles.

**Not executed here.** `FRGCN_Dense` needs the Deep Learning Toolbox, which was
not available in the verification environment. It parses cleanly but has not
been run, so treat it as the least-tested part of this package.

`FRGCN_Dense` trains with `trainNetwork` on 4-D numeric images and a 4-D
categorical label array — the long-stable API for semantic segmentation from
in-memory data. Augmentation for the nnU-Net-style variant is applied offline
into the training array rather than through `augmentedImageDatastore`, which
takes per-image class labels and is the wrong tool for per-pixel label images.
The script tolerates release differences: it falls back gracefully if
`OutputNetwork` or `ClassWeights` are unsupported, and silences the
`unetLayers` deprecation warning (that function still works; it is only a
warning).

**This stage is optional.** The dense baselines are an honest anchor for the
absolute numbers, not part of the paper's claim. If the Deep Learning Toolbox
gives you trouble, skip `FRGCN_Dense` entirely — every other table is still
produced, and Table 6 simply omits the U-Net rows.

## Expected numbers

MATLAB and NumPy draw from different random streams, so weight initialisation,
shuffling and the bootstrap will not match bit for bit. Results should agree
with the published tables **within seed variation** (the paper reports a
seed-to-seed SD of roughly 0.011-0.021 Dice), not exactly. The qualitative
conclusions — which component is load-bearing, the direction and significance
of each comparison — are what should reproduce.

## Runtime

The full protocol is 5 folds x 5 seeds x 9 configurations x up to 150 epochs
over ~170 training graphs per fold: roughly 5.7 million graph updates. Expect
**16-24 hours** on a typical desktop; MATLAB's interpreter, not the algorithm,
is the bottleneck (the Python reference does the same work in about 80 minutes
because it vectorises the same loops through NumPy).

To reduce it:

- `cfg.seeds = 0:2` instead of `0:4` — about 40% less work, and the seed SD is
  still estimable from three runs
- `parfor` over `f = 1:cfg.kFolds` in `FRGCN_RunAll` if you have Parallel
  Computing Toolbox — the folds are independent
- `FRGCN_QUICK = true` for a correctness check rather than a result

Feature extraction is a one-off ~15-20 minutes and is cached in
`cache_graphs.mat`.
