Mahmoud Sayed
210201886

# WEX 428 - Deep Image Prior-Based Image Restoration

Computer Vision course project implementing **Deep Image Prior (DIP)** for single-image denoising and deblurring in Python and PyTorch without pre-trained weights or external restoration APIs.

## Project Structure

- `input_module.py` — **Image Input Module**: Prepares the 2,500-image dataset (2,000 training/optimization images and 500 reserved test images) and assigns random per-image degradation parameters in `data/dataset_manifest.json`.
- `degradation_module.py` — **Image Degradation Module**: Applies Gaussian noise, Salt-and-Pepper noise, Gaussian blur, and Motion blur.
- `dip_models.py` — **Deep Image Prior Reconstruction Module**: Implements `UNetDIP` (with skip connections) and `EncoderDecoderDIP` (without skip connections) in PyTorch.
- `denoising_module.py` — **Denoising Module**: Single-image DIP optimization loop for noise removal with input perturbation and EMA smoothing.
- `deblurring_module.py` — **Deblurring Module**: Single-image DIP optimization loop using known blur kernel convolution and Total Variation (TV) regularization.
- `evaluation_module.py` — **Evaluation Module**: Computes MSE, PSNR (dB), and SSIM metrics.
- `visualization_module.py` — **Visualization Module**: Generates side-by-side comparison figures, iteration snapshots, and metric convergence plots.
- `experiment_module.py` — **Experiment Comparison Module**: Evaluates different noise levels, blur kernels, network architectures, and early stopping points, saving outputs to `results/`.
- `app.py` — **Live Testing Interface**: Interactive Streamlit interface operating strictly on the held-out 500-image test subset for the live defense demonstration.

## Setup and Installation

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Optional for Windows AMD Radeon GPUs such as RX 5700 XT)*:
   ```bash
   pip install torch-directml
   ```

2. Prepare the 2,500-image dataset and run the comparison experiments:
   ```bash
   python experiment_module.py
   ```

3. Launch the Live Testing Interface for the defense demo:
   ```bash
   streamlit run app.py
   ```
