# VAE Assignment

PyTorch notebook comparing latent-space dimensions, model architectures, and **generated sample quality** using a 2D two-moons VAE and an MNIST conditional VAE.

## Setup (VS Code)

1. Create and activate the Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate vae-clean
   ```

2. Register the Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name vae-clean --display-name "Python (VAE Clean)"
   ```

3. Open `VAE_Assignment_Experiments.ipynb` in VS Code and select **Python (VAE Clean)** as the kernel.

4. Run the notebook cells in order. Use `QUICK_RUN = True` for testing and `QUICK_RUN = False` for the full experiments.

## Experiments

- **Two Moons VAE:** Compare latent dimensions and network architectures on a 2D synthetic dataset.
- **MNIST Conditional VAE:** Generate handwritten digits conditioned on class labels.
- **Evaluation:** Compare reconstruction loss, KL divergence, and generated sample quality across configurations.

Generated plots and experimental results are saved in `vae_outputs/`.

## Experiments

- **Two moons:** latent dimensions 1, 2, 4, 8; shallow, medium, deep networks; loss metrics and plots of *new generated points*.
- **MNIST conditional VAE:** latent dimensions 2, 8, 16, 32; shallow, medium, deep networks; reconstruction/KL losses and *new class-conditioned digits*.
- A final cell compares real images, reconstructions, and new samples. Figures are saved to `vae_outputs/`.

Set `QUICK_RUN = True` for an initial short test. Set it to `False`, restart the kernel, and run all cells for report results. MNIST downloads once into `data/` (internet required for the first download). Review the final Markdown prompts and write conclusions based on your actual outputs.

**If `vae-assignment` already exists and you want a clean rebuild:** `conda env remove -n vae-assignment` before the `conda env create` command. Close VS Code before removing an active environment.
