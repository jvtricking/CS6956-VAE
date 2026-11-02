# Variational Autoencoder (VAE) Assignment

A reproducible PyTorch notebook exploring **latent-space dimensionality, model architecture, and generated-sample quality**. The assignment begins with 2D synthetic **Two Moons** data and then uses a **conditional VAE (CVAE)** to generate labeled MNIST handwritten digits.

## Files

- `VAE_Assignment_Experiments.ipynb` — implementation, experiments, plots, and report-writing prompts.
- `environment.yml` — Conda environment for Python, Jupyter, and the required libraries.
- `vae_outputs/` — generated automatically when the notebook runs; contains figures and CSV tables.
- `data/` — created when MNIST downloads on first use.

## Setup (Conda)

Run these commands from the folder containing the notebook and `environment.yml`:

```bash
conda env create -f environment.yml
conda activate vae-assignment
python -m ipykernel install --user --name vae-assignment --display-name "Python (vae-assignment)"
jupyter lab
```

In JupyterLab, open `VAE_Assignment_Experiments.ipynb`, choose the **Python (vae-assignment)** kernel, then run the cells in order.

**Updating an existing environment:**

```bash
conda env update -f environment.yml --prune
conda activate vae-assignment
```

**Sanity-check imports:**

```bash
python -c "import torch, torchvision, sklearn, pandas, matplotlib; print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__, '| CUDA:', torch.cuda.is_available())"
```

The notebook automatically runs on CUDA when PyTorch detects a supported GPU, or CPU otherwise. The pip-installed PyTorch builds may not enable GPU acceleration on every OS/GPU setup. For NVIDIA CUDA-specific installation instructions, use the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) and keep `torch` and `torchvision` on compatible releases.

## Running the experiments

In the first code cell, use:

```python
QUICK_RUN = True
```

**First pass:** With `QUICK_RUN=True`, run all cells to verify the environment, data download, training, and figures. This uses fewer latent configurations and shorter training (Two Moons: 8 epochs; MNIST: 3 epochs) with smaller MNIST subsets. These outputs are only smoke tests, **not final results**.

**Full comparison:** Set `QUICK_RUN=False`, restart the kernel, and run all cells again. This enables:

| Experiment | Default configurations | Epochs |
|---|---|---|
| Two Moons latent sweep | 1, 2, 4, 8 | 45 |
| Two Moons architecture | 1×32, 2×64, 3×128 (latent size 2) | 45 |
| MNIST CVAE latent sweep | 2, 8, 16, 32 | 25 |
| MNIST CVAE architecture | 1×64, 128→64, 256→128→64 (latent size 8) | 25 |
| Optional β-VAE | β = 0.1, 1, 4 (latent size 8) | 25 |

The notebook defaults to **12,000 MNIST training / 2,000 validation / 2,000 test** examples in full mode for practical runtime. In quick mode it uses 2,000 / 500 / 500. To experiment with the entire MNIST dataset, change `MNIST_TRAIN_LIMIT`, `MNIST_VAL_LIMIT`, and `MNIST_TEST_LIMIT` to `None` in the MNIST setup cell; the original training set is split into training and validation before limiting. Full runs train multiple models and may take substantially longer on CPU.

The β experiment is enabled by default only when `QUICK_RUN=False`. You can skip it by setting `RUN_BETA_EXPERIMENT = False` in that section.

## What the notebook measures

The VAE encoder learns latent mean and log variance, uses the **reparameterization trick**, and the decoder reconstructs samples. Training minimizes a reconstruction term plus a β-weighted KL-divergence term. The Two Moons and MNIST sections use different reconstruction objectives suited to their data; **do not directly compare their numeric losses**.

- **Train/validation curves and held-out test metrics**: reconstruction, KL divergence, and VAE objective.
- **Generated synthetic points**: sample a new latent vector from the prior and see whether both moon-shaped modes are represented.
- **MNIST generations**: sample from the prior, conditioning on a chosen digit label; examine recognizability and diversity.
- **MNIST reconstructions**: compare held-out images to their reconstructed versions, separately from new generations.
- **Architecture/latent ablations**: change one main factor at a time to make comparisons interpretable.

Results and figures are saved under `vae_outputs/`, including files such as `synthetic_generated_by_latent.png`, `mnist_generated_z8.png`, `mnist_latent_validation.csv`, and `mnist_latent_test.csv` (names depend on enabled experiments).

## Recommended analysis for your report

1. **Primary result — latent dimensions:** Compare 1/2/4/8 for Two Moons and 2/8/16/32 for MNIST. Discuss reconstruction, KL, and whether generated samples visibly improve.
2. **Sample quality beyond loss:** On Two Moons, look for coverage of both arcs and unwanted points between them. On MNIST, inspect digit identity, handwriting variety, blur, and repeated-looking samples.
3. **Architecture effects:** Hold latent size fixed and compare depth/width, including model parameter counts.
4. **Optional β study:** Discuss reconstruction detail versus prior regularization. Avoid treating losses with different β values as directly comparable weighted objectives; inspect the separate reconstruction and KL terms.
5. **Limitations:** Visual inspection is subjective; results use one seed and default MNIST subsets unless modified. Report actual settings and sampled outputs rather than assumed trends.

**Methodological note:** Use validation results to compare/select settings, then report the separate test-set results at the end. Sample quality from the prior is distinct from reconstruction quality; the professor explicitly asks for both numerical analyses and generated samples.

## Troubleshooting

- **`ModuleNotFoundError`**: Ensure Jupyter is using the `Python (vae-assignment)` kernel, rather than another Conda environment.
- **`torchvision` import failure**: Recreate/update the environment; PyTorch 2.5.1 pairs with torchvision 0.20.1 in this file.
- **MNIST download fails**: Internet access is needed for the first MNIST run; rerun once network access is available. Subsequent runs use the local `data/` directory.
- **Training is slow or runs out of memory**: Leave `QUICK_RUN=True` to test; reduce MNIST sample limits, batch sizes, or epochs; skip the optional β experiment.
- **Different results on reruns**: Random seeds are set, but GPU operations and library/platform differences may still introduce variation.

## Suggested submission

Submit the notebook, `README.md`, `environment.yml`, and any key figures/tables required by the instructor. Execute the full experiment settings and ensure your written analysis discusses **actual generated samples** and the effects of **latent dimensionality**.
