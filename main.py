import json

import yaml
import pandas as pd

from src.visualization.plots import plot_interactive
from src.core.fit import TGSAnalyzer

if __name__ == '__main__':
    with open('config.yaml', "r") as file:
        config = yaml.safe_load(file)

    TGSAnalyzer.run_analysis(config)

    with open('data/fit/signal.json', 'r') as file: signal = json.load(file)
    fit = pd.read_csv('data/fit/fit.csv')
    plot_interactive(signal, fit)
