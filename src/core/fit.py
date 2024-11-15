import json
from typing import Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis.tgs import tgs_fit
from src.core.utils import get_num_signals, get_file_prefix
from src.core.path import Paths

class TGSAnalyzer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        base_path = Path(config['path'])
        self.paths = Paths(
            data_dir=base_path,
            figure_dir=base_path / 'figures',
            fit_dir=base_path / 'fit',
            fit_path=base_path / 'fit' / 'fit.csv',
            signal_path=base_path / 'fit' / 'signal.json',
        )
        self.paths.fit_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figure_dir.mkdir(parents=True, exist_ok=True)
        self.idxs = config['idxs']

    def fit_signal(self, file_idx: int, pos_file: str, neg_file: str) -> Tuple[pd.DataFrame, List[List[float]], List[List[float]]]:
        (start_idx, start_time, grating_spacing, 
         A, A_err, B, B_err, C, C_err, 
         alpha, alpha_err, beta, beta_err, 
         theta, theta_err, tau, tau_err, 
         f, f_err, signal) = tgs_fit(self.config, self.paths, file_idx, pos_file, neg_file, **self.config['tgs'])

        params = {
            'A': (A, A_err, 'Wm^-2'),
            'B': (B, B_err, 'Wm^-2'),
            'C': (C, C_err, 'Wm^-2'),
            'alpha': (alpha, alpha_err, 'm^2s^-1'),
            'beta': (beta, beta_err, 's^0.5'),
            'theta': (theta, theta_err, 'rad'),
            'tau': (tau, tau_err, 's'),
            'f': (f, f_err, 'Hz'),
        }

        data = {
            'run_name': Path(pos_file).name,
            'start_idx': start_idx,
            'start_time': start_time,
            'grating_value[µm]': grating_spacing,
            **{f'{name}[{unit}]': value for name, (value, _, unit) in params.items()},
            **{f'{name}_err[{unit}]': error for name, (_, error, unit) in params.items()},
        }

        return pd.DataFrame([data]), signal.tolist()

    def fit(self) -> None:
        fit_data = pd.DataFrame()
        signals = []

        if self.idxs is None:
            num_signals = get_num_signals(self.paths.data_dir)
            self.idxs = range(1, num_signals + 1)

        for i in self.idxs:
            print(f"Analyzing signal {i}")
            if not (file_prefix := get_file_prefix(self.paths.data_dir, i)):
                print(f"Could not find file prefix for signal {i}")
                continue
            # TODO: plot raw signal under failure case
            pos_file = self.paths.data_dir / f'{file_prefix}-POS-{i}.txt'
            neg_file = self.paths.data_dir / f'{file_prefix}-NEG-{i}.txt'

            try:
                df, signal = self.fit_signal(i, pos_file, neg_file)
                signals.append(signal)
                fit_data = pd.concat([fit_data, df], ignore_index=True)
            except Exception as e:
                print(f"Error fitting signal {i}: {e}")
                continue

        fit_data.to_csv(self.paths.fit_path, index=False)
        with open(self.paths.signal_path, 'w') as f: json.dump(signals, f)

        # TODO: fit summary
