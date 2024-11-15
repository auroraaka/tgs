# Transient Grating Spectroscopy Analysis Tool

A Python tool for analyzing transient grating spectroscopy (TGS) data. Process raw signals and extract thermo-mechanical properties through automated analyses and curve fitting.

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/auroraaka/tgs.git
   cd tgs
   ```

2. Choose one of the following installation methods:

   **Option 1:** Using setup.py
   ```bash
   pip install -e .
   ```

   **Option 2:** Using requirements.txt
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Edit `config.yaml` to set your data path and desired fitting parameters.

2. Run Analysis
   ```bash
   python main.py
   ```
   Fitting results and figures will be saved in `fit/` and `figures/` directories, respectively.
