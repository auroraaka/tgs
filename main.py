import json

import yaml
import pandas as pd

from src.core.fit import TGSAnalyzer
from src.core.plots import plot_interactive

if __name__ == '__main__':

    with open('config.yaml', "r") as file: config = yaml.safe_load(file)
    analyzer = TGSAnalyzer(config)
    analyzer.fit()

    # with open(analyzer.paths.signal_path, 'r') as file: signal = json.load(file)
    # fit = pd.read_csv(analyzer.paths.fit_path)
    # plot_interactive(signal, fit)
