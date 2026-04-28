# Galaxy Morphology Classification — Companion Code

This repository accompanies the paper:

> **Razeghi, A. (2025).** *Machine Learning and AI in Modern Astronomy: A Review of Methods, Applications, Challenges, and a Reproducible Case Study.*

It contains the full reproducible pipeline for the case study presented in Section 5 of the paper, including:

- A synthetic galaxy dataset generator (three morphological classes: Elliptical, Spiral, Irregular)
- Two lightweight baseline classifiers (HOG + Random Forest, and an MLP on raw pixels)
- A PyTorch implementation of a small Convolutional Neural Network
- All figures used in the paper

## Quick start

```bash
# Clone and install
git clone https://github.com/<your-username>/galaxy-ml-case-study.git
cd galaxy-ml-case-study
pip install -r requirements.txt

# Open the notebook
jupyter notebook galaxy_morphology_case_study.ipynb
```

Total runtime for the baselines on a single CPU is under 20 seconds. The optional PyTorch CNN trains in a few minutes on CPU or seconds on GPU.

## Reproducing the paper results

| Model | Test accuracy | Train time |
|---|---|---|
| Random Forest (HOG features) | 0.989 | ~3 s |
| MLP (raw pixels, 256–128 ReLU) | 0.967 | ~11 s |
| CNN (PyTorch, optional) | ~0.99+ | minutes (CPU) / seconds (GPU) |

All random seeds are fixed at 42 for exact reproducibility of these numbers.

## Replacing the synthetic dataset with real Galaxy10 SDSS imagery

The synthetic generator keeps the notebook fully self-contained. To repeat the experiment on real survey data, install [astroNN](https://astronn.readthedocs.io/) and replace the dataset call:

```python
from astroNN.datasets import load_galaxy10
X, y = load_galaxy10()  # ~1.4 GB download
```

## Repository layout

```
.
├── galaxy_morphology_case_study.ipynb   # Main notebook (recommended entry point)
├── generate_dataset.py                  # Standalone dataset generator
├── train_models.py                      # Standalone training script for baselines
├── figures/                             # Pre-generated figures used in the paper
├── results_summary.json                 # Numerical results recorded by train_models.py
├── requirements.txt
└── README.md
```

## Requirements

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
scikit-image>=0.21
matplotlib>=3.7
torch>=2.0          # optional, only for the CNN section
```

## Citation

If you use this code in published work, please cite the accompanying paper.

## License

MIT — see LICENSE for details.

## Contact

Ali Razeghi — Newmarket, Ontario, Canada
