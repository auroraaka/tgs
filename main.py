import os

import yaml

from plots import plot_interactive, plot_time_vs_response
from fit import TGSAnalyzer

if __name__ == '__main__':
    with open('config.yaml', "r") as file:
        config = yaml.safe_load(file)

    TGSAnalyzer.run_analysis(config)

    # dir_path = os.path.join(config['path'], 'fit')
    # fit_data, x_raw, y_raw, x_fit, y_fit =
    #     os.path.join(dir_path, 'fit.csv'),
    #     os.path.join(dir_path, 'x_raw.json'),
    #     os.path.join(dir_path, 'y_raw.json'),
    #     os.path.join(dir_path, 'x_fit.json'),
    #     os.path.join(dir_path, 'y_fit.json')
    # idx = 2
    # plot_time_vs_response(fit_data, x_raw, y_raw, x_fit, y_fit, idx)

    # plot_interactive(postfit, prefit, x_raw, y_raw, x_fit, y_fit)