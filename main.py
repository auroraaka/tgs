import yaml

from src.visualization.plots import plot_interactive
from src.core.fit import TGSAnalyzer

if __name__ == '__main__':
    with open('config.yaml', "r") as file:
        config = yaml.safe_load(file)

    TGSAnalyzer.run_analysis(config)

    # plot_interactive(postfit, prefit, x_raw, y_raw, x_fit, y_fit)