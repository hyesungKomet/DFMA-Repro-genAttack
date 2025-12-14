# Data-Free Model-Related Attacks Unleashing the Potential of Generative AI Replication & Extension
## About
This repository contains the source code for data-free model-related attacks that leverage large generative models to replicate the data distribution of smaller victim models. We implement attacks on three benchmark tasks: MNIST, CIFAR10, and SKIN_CANCER.

This work is based on and extends the original code released at [Zenodo](https://zenodo.org/records/14737003).

## Attack Intuition
To validate our approach, we compare the distribution scatter plots of the raw training data and the generated data. Using CIFAR10 as an example, the distributions are remarkably similar, demonstrating the effectiveness of our attack strategy.

<p align="center">
    <img src="./generated_demo/compare.png" alt="Distribution Comparison" width="1000" height="420">
</p>

## Project Structure
Each task folder (e.g., `cifar10/`, `mnist/`, `skin_cancer/`, `text/`) contains a detailed `Readme.md` with implementation steps, scripts, and usage instructions.

## Demo
We provide a Gradio-based demo for visualizing the attacks on CIFAR10. After training the three attack models (extraction, membership inference, and inversion), set the model paths in the demo script.

### Usage
1. Train the required models for CIFAR10.
2. Update the model paths in `app_demo.py`.
3. Run the demo:
   ```bash
   python app_demo.py
   ```
4. Input either an original CIFAR10 sample or a synthetic sample. The demo will perform inference using the victim model, extracted model, membership inference attack, and model inversion attack, displaying the results.

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- Gradio
- Other dependencies as listed in each task's `Readme.md`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hyesungKomet/DFMA-Repro-genAttack.git
   cd DFMA-Repro-genAttack
   ```
2. Install dependencies (refer to individual task folders for specific requirements).

## Contributing
Contributions are welcome. Please follow the guidelines in each task's `Readme.md` and ensure code quality.