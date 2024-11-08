# Transient Grating Spectroscopy Analysis Tool

A repository designed for analyzing transient grating spectroscopy data. This tool enables processing raw signals and extracting thermo-mechanical material properties through analysis and fitting techniques.

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

1. Configure Analysis Parameters
   - Edit `config.yaml` to set your desired parameters based on your experiment requirements

2. Data Preparation
   - Place raw data files in `data/raw/`
   - Ensure files match the expected TGS experiment format (e.g. `*.txt` files from PI3)

3. Run Analysis
   ```bash
   python main.py
   ```
   Processed signals and fitting results will be saved in `data/fit/`.

## Repository Structure

```
tgs/
├── data/
│   ├── raw/                # Raw experimental data
│   └── fit/                # Analysis outputs
├── figures/
│   ├── process/            # Processed signal figures
│   └── fft-lorentzian/     # FFT and Lorentzian fit figures
│   └── tgs/                # TGS fit figures
├── src/                    
│   ├── analysis/           
│   │   ├── fft.py          # FFT analysis
│   │   ├── lorentzian.py   # Lorentzian fitting
│   │   ├── tgs.py          # TGS fitting
│   ├── core/               
│   │   ├── fit.py          # Fitting execution
│   │   ├── process.py      # Signal processing
│   ├── utils/              
│   │   ├── utils.py        # Utility functions
│   ├── visualization/      
│   │   ├── plots.py        # Plotting functions
├── tests/                  
├── .gitignore              # Git ignore
├── config.yaml             # Configuration
├── LICENSE.txt             # License
├── main.py                 # Main script
├── README.md               # README file
├── requirements.txt        # Dependencies
└── setup.py                # Installation script
```
