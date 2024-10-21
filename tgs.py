import os
import re
from typing import Any, List, Tuple

import matlab.engine
import numpy as np
import pandas as pd
import scipy

from utils import save_csv, save_json, tgs_function

class TGS:
    def __init__(self, config: dict[str, Any]) -> None:
        self.grating = config['grating']
        self.path = config['path']
        self.raw_path = os.path.join(self.path, 'raw')
        self.fit_path = os.path.join(self.path, 'fit')
        self.prefit_path = os.path.join(self.fit_path, 'prefit.csv')
        self.postfit_path = os.path.join(self.fit_path, 'postfit.csv')
        self.x_raw_path = os.path.join(self.fit_path, 'x_raw.json')
        self.y_raw_path = os.path.join(self.fit_path, 'y_raw.json')
        self.x_fit_path = os.path.join(self.fit_path, 'x_fit.json')
        self.y_fit_path = os.path.join(self.fit_path, 'y_fit.json')

        self.eng = matlab.engine.start_matlab()
        self.eng.cd(os.getcwd())

    def tgs_phase_analysis(self, pos_file: str, neg_file: str,
                           pos_baseline: float, neg_baseline: float, verbose: int = 0) -> Tuple[pd.DataFrame, List[float], List[float]]:
        run_name = os.path.basename(pos_file)
        output = self.eng.TGSPhaseAnalysis(pos_file, neg_file, self.grating, 2, 0, 0, pos_baseline, neg_baseline,
                                           '', verbose, 16, 0, 0, nargout=22)

        freq, freq_err, speed, alpha, alpha_err, tau, tau_err, A, A_err, beta, beta_err, B, B_err, theta, theta_err, C, C_err, start_time, x_raw, y_raw, x_fit, y_fit = output
        tau = np.asarray(tau)[0][2]
        x_raw = list(np.array(x_raw).squeeze(1))
        y_raw = list(np.array(y_raw).squeeze(1))
        x_fit = list(np.array(x_fit).squeeze(1))
        y_fit = list(np.array(y_fit).squeeze(1))

        data = {
            'run_name': run_name,
            'grating_value[um]': self.grating,
            'SAW_freq[Hz]': freq,
            'SAW_freq_error[Hz]': freq_err,
            'SAW_speed[m/s]': speed,
            'A[Wm^-2]': A,
            'A_err[Wm^-2]': A_err,
            'alpha[m^2s^-1]': alpha,
            'alpha_err[m^2s^-1]': alpha_err,
            'beta[s^0.5]': beta,
            'beta_err[s^0.5]': beta_err,
            'B[Wm^-2]': B,
            'B_err[Wm^-2]': B_err,
            'theta': theta,
            'theta_err': theta_err,
            'tau[s]': tau,
            'tau_err[s]': tau_err,
            'C[Wm^-2]': C,
            'C_err[Wm^-2]': C_err,
            'start_time': start_time,
        }

        return pd.DataFrame([data]), x_raw, y_raw, x_fit, y_fit

    def get_num_signals(self) -> int:
        pattern = re.compile(r'POS-(\d+)\.txt')
        return max((int(match.group(1)) for filename in os.listdir(self.raw_path)
                    if (match := pattern.search(filename))), default=0)

    def get_file_prefix(self, i: int) -> str:
        pattern = re.compile(rf'(.+)-POS-{i}\.txt')
        for filename in os.listdir(self.raw_path):
            if match := pattern.match(filename):
                return match.group(1)
        return None

    def fit(self, idxs: List[int] = None) -> None:
        prefit_data = pd.DataFrame()
        postfit_data = pd.DataFrame()
        x_raw, y_raw, x_fit, y_fit = [], [], [], []

        if idxs is None:
            num_signals = self.get_num_signals()
            idxs = range(1, num_signals + 1)

        for i in idxs:
            print(f"Analyzing signal {i}")

            file_prefix = self.get_file_prefix(i)
            if not file_prefix:
                print(f"Could not find file prefix for signal {i}")
                continue

            pos_file = os.path.join(self.raw_path, f'{file_prefix}-POS-{i}.txt')
            neg_file = os.path.join(self.raw_path, f'{file_prefix}-NEG-{i}.txt')

            try:
                df, xr, yr, xf, yf = self.tgs_phase_analysis(pos_file, neg_file, 0, 0)

                x_raw.append(xr)
                y_raw.append(yr)
                x_fit.append(xf)
                y_fit.append(yf)

                prefit = df.iloc[0].to_dict()
                arg_keys = ['A[Wm^-2]', 'B[Wm^-2]', 'C[Wm^-2]', 'alpha[m^2s^-1]', 'beta[s^0.5]', 'theta', 'tau[s]', 'SAW_freq[Hz]']
                arg_err_keys = ['A_err[Wm^-2]', 'B_err[Wm^-2]', 'C_err[Wm^-2]', 'alpha_err[m^2s^-1]', 'beta_err[s^0.5]', 'theta_err', 'tau_err[s]', 'SAW_freq_error[Hz]']
                arg_vals = [float(prefit[key]) for key in arg_keys]

                start_time = float(prefit['start_time'])
                tgs, _ = tgs_function(start_time, self.grating)
                fit, cov = scipy.optimize.curve_fit(tgs, xf, yf, p0=arg_vals, maxfev=10000)
                err = np.sqrt(np.diag(cov))

                postfit = prefit.copy()
                for key, value in zip(arg_keys, fit):
                    postfit[key] = value
                for key, value in zip(arg_err_keys, err):
                    postfit[key] = value

                for prefix in ['prefit', 'postfit']:
                    data = locals()[prefix]
                    data['SAW_speed[m/s]'] = data['SAW_freq[Hz]'] * data['grating_value[um]'] * 1e-6
                    data['SAW_speed_error[m/s]'] = data['SAW_freq_error[Hz]'] * data['grating_value[um]'] * 1e-6

                prefit_data = pd.concat([prefit_data, pd.DataFrame([prefit])], ignore_index=True)
                postfit_data = pd.concat([postfit_data, pd.DataFrame([postfit])], ignore_index=True)

            except Exception as e:
                print(f"Could not successfully analyze signal {i}, Error: {e}")

        save_csv(prefit_data, self.prefit_path)
        save_csv(postfit_data, self.postfit_path)
        save_json(x_raw, self.x_raw_path)
        save_json(y_raw, self.y_raw_path)
        save_json(x_fit, self.x_fit_path)
        save_json(y_fit, self.y_fit_path)

    @classmethod
    def run_analysis(cls, config: dict[str, Any], idxs: List[int] = None) -> None:
        tgs = cls(config)
        tgs.fit(idxs)
        tgs.eng.quit()