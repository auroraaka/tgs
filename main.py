import json

import yaml
import pandas as pd

from src.visualization.plots import plot_interactive
from src.core.fit import TGSAnalyzer

if __name__ == '__main__':
    with open('config.yaml', "r") as file: config = yaml.safe_load(file)
    analyzer = TGSAnalyzer(config)
    analyzer.run_analysis(config)

    # with open(analyzer.paths.signal, 'r') as file: signal = json.load(file)
    # fit = pd.read_csv(analyzer.paths.fit)
    # plot_interactive(signal, fit)
