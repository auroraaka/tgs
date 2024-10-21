import os

from plots import plot_fits_interactive, plot_time_vs_response
from tgs import TGS
from utils import load_config, load_data


if __name__ == '__main__':
    config = load_config('config.yaml')

    # TGS.run_analysis(config['tgs']['cooldown'])
    # TGS.run_analysis(config['tgs']['irradiation'])

    dir_path = os.path.join(config['tgs']['cooldown']['path'], 'fit')
    postfit, prefit, x_raw, y_raw, x_fit, y_fit = load_data(
        os.path.join(dir_path, 'postfit.csv'),
        os.path.join(dir_path, 'prefit.csv'),
        os.path.join(dir_path, 'x_raw.json'),
        os.path.join(dir_path, 'y_raw.json'),
        os.path.join(dir_path, 'x_fit.json'),
        os.path.join(dir_path, 'y_fit.json')
    )
    idx = 2
    plot_time_vs_response(postfit, prefit, x_raw, y_raw, x_fit, y_fit, idx)

    # plot_fits_interactive(postfit, prefit, x_raw, y_raw, x_fit, y_fit)