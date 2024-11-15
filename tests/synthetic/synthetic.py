import json
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

from src.core.fit import TGSAnalyzer
from src.analysis.functions import tgs_function

def generate_signal(params, time_range=(1e-12, 2e-7), num_points=4000, noise_level=0.1): # modulate noise, types of noise
    """
    Generate synthetic TGS signal with optional noise.
    
    Parameters:
        params (dict): Dictionary containing the TGS parameters:
            - A: thermal signal amplitude [W/m²]
            - B: acoustic signal amplitude [W/m²]
            - C: signal offset [W/m²]
            - alpha: thermal diffusivity [m²/s]
            - beta: displacement-reflectance ratio [s⁰⋅⁵]
            - theta: acoustic phase [rad]
            - tau: acoustic decay time [s]
            - f: surface acoustic wave frequency [Hz]
            - grating_spacing: grating spacing [µm]
        time_range (tuple): (start_time, end_time) in seconds
        num_points (int): Number of time points to generate
        noise_level (float): Standard deviation of Gaussian noise to add
        
    Returns:
        np.ndarray: Array of shape (N, 2) containing [time, amplitude]
    """
    t = np.linspace(time_range[0], time_range[1], num_points)
    functional_fit, _ = tgs_function(start_time=2e-9, grating_spacing=params['grating_spacing'])
    signal = functional_fit(t, params['A'], params['B'], params['C'], params['alpha'], params['beta'], params['theta'], params['tau'], params['f'])
    
    noise = np.random.normal(0, noise_level * np.std(signal), num_points)
    noisy_signal = signal + noise

    return np.column_stack((t, noisy_signal))

def generate_test_cases():
    test_cases = []
    
    # Test case 1: Typical values
    params1 = {
        'A': 0.0005, # [-1e-2, 1e-2]
        'B': 0.1, # [-1.0, 1.0]
        'C': 0, # [-0.1, 0.1]
        'alpha': 5e-6, # [0, 3.3e-4]
        'beta': 0.05, # [0, 1]
        'theta': np.pi/4, # [-pi, pi]
        'tau': 2e-9, # [0.5e-9, 200e-9]
        'f': 5e8, # [1e6, 50e9]
        'grating_spacing': 3.5 # [100e-3, 50]
    }
    params2 = {
        'A': 0.0003, # [-1e-2, 1e-2]
        'B': 0.08, # [-1.0, 1.0]
        'C': 0.1, # [-0.1, 0.1]
        'alpha': 5e-6, # [0, 3.3e-4]
        'beta': 0.05, # [0, 1]
        'theta': np.pi/4, # [-pi, pi]
        'tau': 2e-9, # [0.5e-9, 200e-9]
        'f': 5.1e8, # [1e6, 50e9]
        'grating_spacing': 3.5 # [100e-3, 50]
    }
    pos_signal1 = generate_signal(params1)
    neg_signal1 = generate_signal(params2)
    test_cases.append((pos_signal1, neg_signal1, params1))
    
    # Test case 2: High thermal diffusivity
    params2 = {**params1, 'alpha': 1e-5, 'A': 0.07}
    pos_signal2 = generate_signal(params2)
    neg_signal2 = -pos_signal2
    test_cases.append((pos_signal2, neg_signal2, params2))
    
    # Test case 3: Strong acoustic component
    params3 = {**params1, 'B': 0.08, 'f': 7e8}
    pos_signal3 = generate_signal(params3)
    neg_signal3 = -pos_signal3
    test_cases.append((pos_signal3, neg_signal3, params3))
    
    return test_cases

def save_signal_file(signal: np.ndarray, filepath: Path, metadata: dict) -> None:

    header = (
        f"Study Name\t{metadata['study_name']}\n"
        f"Sample Name\t{metadata['sample_name']}\n"
        f"Run Name\t{metadata['run_name']}\n"
        f"Operator\tsynthetic\n"
        f"Date\t{metadata['date']}\n"
        f"Time\t{metadata['time']}\n"
        f"Sign\t{metadata['sign']}\n"
        f"Grating Spacing\t{metadata['grating_spacing']}um\n"
        f"Channel\t3\n"
        f"Number Traces\t10000\n"
        f"Files in Batch\t1\n"
        f"Batch Number\t{metadata['batch']}\n"
        f"dt\t50.000001E-12\n"
        f"time stamp (ms)\t12:00:00 PM\n"
        "\n"
        "Time\tAmplitude\n"
        ""
    )
    
    with open(filepath, 'w', newline='') as f:
        f.write(header)
        for time, amplitude in signal:
            f.write(f"{time:.6E}\t{amplitude:.6E}\n")

def save_test_cases(test_cases, base_dir='tests/synthetic/data'):
    data_dir = Path(base_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    for i, (pos_signal, neg_signal, params) in enumerate(test_cases, 1):
        test_dir = data_dir / f'test-{i}'
        test_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            'study_name': f'Synthetic-{i}',
            'sample_name': f'Test-{i}',
            'run_name': 'validation',
            'date': current_date,
            'time': current_time,
            'grating_spacing': f"{params['grating_spacing']:05.2f}",
            'batch': 1
        }

        metadata['sign'] = 'POS'
        pos_path = test_dir / f'synthetic-POS-1.txt'
        save_signal_file(pos_signal, pos_path, metadata)
        
        metadata['sign'] = 'NEG'
        neg_path = test_dir / f'synthetic-NEG-1.txt'
        save_signal_file(neg_signal, neg_path, metadata)

        with open(test_dir / 'true_params.json', 'w') as f:
            json.dump(params, f, indent=4)

def test(test_idx: int, config_path='tests/synthetic/config.yaml'):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['path'] = Path(f'tests/synthetic/data/test-{test_idx}')
    
    analyzer = TGSAnalyzer(config)
    analyzer.fit()
    
    fit_results = pd.read_csv(analyzer.paths.fit_path)
    true_params = json.load(open(analyzer.paths.data_dir / 'true_params.json'))

    print(f"\nTest Case {test_idx}:")
    print("Parameter  |  True Value  |  Fitted Value  |  Relative Error (%)")
    print("-" * 60)
        
    param_keys = ['A', 'B', 'C', 'alpha', 'beta', 'theta', 'tau', 'f']
    fit_keys = ['A[Wm^-2]', 'B[Wm^-2]', 'C[Wm^-2]', 'alpha[m^2s^-1]', 
                   'beta[s^0.5]', 'theta[rad]', 'tau[s]', 'f[Hz]']
        
    for param, fit_key in zip(param_keys, fit_keys):
        true_val = true_params[param]
        fitted_val = fit_results.iloc[0][fit_key]
        rel_error = abs(fitted_val - true_val) / true_val * 100
        print(f"{param:9} | {true_val:11.3e} | {fitted_val:11.3e} | {rel_error:15.2f}")

if __name__ == '__main__':
    test_cases = generate_test_cases()
    save_test_cases(test_cases)
    for i in range(1, len(test_cases) + 1):
        test(i)