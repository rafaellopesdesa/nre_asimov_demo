# Asimov inference with neural ratio estimation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafaellopesdesa/nre_asimov_demo/blob/main/NRE_Asimov.ipynb)

A standalone demonstration of the finite-sample Asimov construction for neural
ratio estimation (NRE). The notebook generates its own events, trains the
preselection and ratio classifiers, and compares expected likelihood scans and
asymptotic predictions with pseudo-experiments. No pretrained models or other
analysis repositories are required.

## Run the notebook

Open [NRE_Asimov.ipynb](NRE_Asimov.ipynb) in Colab, select a **GPU runtime**,
and run the cells in order. The setup installs the requirements and, by default,
mounts Google Drive. Persistent outputs go to
`MyDrive/nre_asimov_demo/workspace/`.

For local execution, use Python 3.10 or newer, install a PyTorch build suitable
for your GPU, and start Jupyter from this repository:

```bash
python -m pip install -r requirements.txt jupyterlab
jupyter lab NRE_Asimov.ipynb
```

The local default output directory is `workspace/`. Set
`NRE_ASIMOV_WORKSPACE` before launching Jupyter to choose another directory.
All figures appear inline in the notebook. Plotting functions do not write
images or secondary Python programs; save the executed notebook to retain its
displayed figures.

## Analysis

1. Generate five-dimensional signal/background Gaussian mixtures with detector
   smearing. Train a single preselection classifier, select a threshold targeting
   background-to-signal yield ratio 250, and estimate the selected yields on an
   independent simulation bank. Freeze the selection and yields.
2. Build independent selected signal, background, and equal-mixture reference
   samples. Train separate signal/reference and background/reference ensembles.
3. Compare conventional simulator-weighted Asimov samples with reference samples
   whose ratios and Asimov weights use the same finite-sample normalization.
   Check the fitted signal strength and scan convergence as the integration
   sample grows.
4. Compare the distributions of the fitted signal strength and discovery
   statistic against both NRE-model and independent simulator-bank toys. Check
   the numerical compression against uncompressed fits before the full toy run.
5. Diagnose any simulator-to-model discrepancy using independent expected-score
   estimates and their Monte Carlo uncertainties.

The default `FULL` profile uses five million selected events **per training
class**, four classifiers per ratio, and four 1024-unit Swish hidden layers per
classifier. NAdam runs for 140 epochs, with learning rate decreasing from
`1e-4` to `1e-10`; the final weights are used. Ratio training has no dropout,
weight decay, or early stopping. Validation events are independent; ensemble
members share training banks and differ in initialization and shuffling.
The full comparison uses 100,000 toys from each source.

Set `PROFILE = "SMOKE"` in the configuration cell, or set the environment
variable `NRE_ASIMOV_PROFILE=SMOKE`, for a short CPU-compatible workflow check.
It reduces network sizes, training, integration, and toy statistics. Its results
do not establish physics closure. The full run is computationally substantial
even with a GPU; its settings are exposed in the notebook.

Preselection is trained independently in this repository using the same model
family and selection target as the companion likelihood demonstration. Its
weights, cut, and estimated efficiencies are a fresh numerical realization,
so reproducing particular published curves is not guaranteed. Exact
finite-sample Asimov closure is internal to the normalized NRE model; agreement
with simulator toys remains a separate statistical check.

## Reuse and recovery

Simulation banks, completed classifier members, and toy batches are cached with
configuration and content hashes. Rerunning loads matching completed artifacts.
Preselection training, cut calibration, and yield estimation are cached
separately; changing calibration statistics does not retrain the classifier.
An interruption during a classifier member requires restarting that member,
while completed members remain reusable. Numerical tables are written under
`workspace/artifacts/reports/5m_baseline/<profile>/`.

Use one writer per artifact directory. A lock left by an interrupted process
is reported with its exact path; remove that lock only once its process has
stopped. Incomplete or corrupted completed artifacts raise an error instead of
being overwritten automatically.

## Files

- `NRE_Asimov.ipynb`: the complete analysis and mathematical discussion.
- `utils_distributions.py`: the signal/background model and detector response.
- `utils_preselection.py`: standalone preselection training, calibration, and
  selected-yield estimation.
- `utils_nre.py`: simulator banks, BCE ensemble training, and ratio prediction.
- `utils_nre_inference.py`: likelihood fits, finite-sample Asimov construction,
  validated compression, cached toys, and score diagnostics.
- `utils_nre_plotting.py`: inline Matplotlib figures.

The analysis utilities are adapted from
[`nsbi-lhc-toolkit`](https://github.com/rafaellopesdesa/nsbi-lhc-toolkit), at
revision [`2d617967`](https://github.com/rafaellopesdesa/nsbi-lhc-toolkit/commit/2d617967d3926bb57229a6e44e24969611fa7439).
The runtime is self-contained. The software is distributed under the MIT
license; see [LICENSE.txt](LICENSE.txt).
