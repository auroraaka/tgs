import json
from typing import Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tgs import tgs_fit
from utils import get_num_signals, get_file_prefix

@dataclass
class DataPaths:
    raw_folder: Path
    fit_folder: Path
    fit: Path
    x_raw: Path
    y_raw: Path
    x_fit: Path
    y_fit: Path

class TGSAnalyzer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        base_path = Path(config['path'])
        
        self.paths = DataPaths(
            raw_folder=base_path / 'raw',
            fit_folder=base_path / 'fit',
            fit=base_path / 'fit' / 'fit.csv',
            x_raw=base_path / 'fit' / 'x_raw.json',
            y_raw=base_path / 'fit' / 'y_raw.json',
            x_fit=base_path / 'fit' / 'x_fit.json',
            y_fit=base_path / 'fit' / 'y_fit.json'
        )

    def process_signal(self, pos_file: str, neg_file: str) -> Tuple[pd.DataFrame, List[float], List[float], List[float], List[float]]:

        output = tgs_fit(self.config, pos_file, neg_file, **self.config['tgs'])
        start_time, grating_spacing, A, A_err, B, B_err, C, C_err, alpha, alpha_err, beta, beta_err, theta, theta_err, tau, tau_err, f, f_err, full_signal, truncated_signal = output
        x_raw, y_raw = full_signal[:, 0], full_signal[:, 1]
        x_fit, y_fit = truncated_signal[:, 0], truncated_signal[:, 1]

        params = {
            'A': (A, A_err, 'Wm^-2'),
            'B': (B, B_err, 'Wm^-2'),
            'C': (C, C_err, 'Wm^-2'),
            'alpha': (alpha, alpha_err, 'm^2s^-1'),
            'beta': (beta, beta_err, 's^0.5'),
            'theta': (theta, theta_err, ''),
            'tau': (tau, tau_err, 's'),
            'f': (f, f_err, 'Hz'),
        }

        data = {
            'run_name': Path(pos_file).name,
            'start_time': start_time,
            'grating_value[um]': grating_spacing,
            **{f'{name}[{unit}]': value for name, (value, _, unit) in params.items()},
            **{f'{name}_err[{unit}]': error for name, (_, error, unit) in params.items()},
        }

        return pd.DataFrame([data]), x_raw.tolist(), y_raw.tolist(), x_fit.tolist(), y_fit.tolist()

    def process_all(self, idxs: List[int] = None) -> None:
        fit_data = pd.DataFrame()
        signals = {key: [] for key in ['x_raw', 'y_raw', 'x_fit', 'y_fit']}

        if idxs is None:
            num_signals = get_num_signals(self.paths.raw_folder)
            idxs = range(1, num_signals + 1)

        for i in idxs:
            print(f"Analyzing signal {i}")
            if not (file_prefix := get_file_prefix(self.paths.raw_folder, i)):
                print(f"Could not find file prefix for signal {i}")
                continue

            pos_file = self.paths.raw_folder / f'{file_prefix}-POS-{i}.txt'
            neg_file = self.paths.raw_folder / f'{file_prefix}-NEG-{i}.txt'
            
            df, xr, yr, xf, yf = self.process_signal(pos_file, neg_file)
            for data, key in zip([xr, yr, xf, yf], signals.keys()):
                signals[key].append(data)
            fit_data = pd.concat([fit_data, df], ignore_index=True)

        fit_data.to_csv(self.paths.fit, index=False)
        for key, data in signals.items():
            path = getattr(self.paths, key)
            path.write_text(json.dumps(data))

    @classmethod
    def run_analysis(cls, config: dict[str, Any], idxs: List[int] = None) -> None:
        tgs = cls(config)
        tgs.process_all(idxs)
