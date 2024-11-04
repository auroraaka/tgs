import matplotlib
import numpy as np

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from pathlib import Path

from src.analysis.functions import tgs_function

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']

NUM_POINTS = 1000


def plot_tgs(signal, start_idx, functional_function, thermal_function, fit_params, idx):
    x_raw, y_raw = signal[:NUM_POINTS, 0], signal[:NUM_POINTS, 1]
    x_fit = signal[start_idx:NUM_POINTS, 0]

    plt.figure(figsize=(10, 6))
    plt.plot(x_raw * 1e9, y_raw * 1e3, linestyle='-', color='black', linewidth=2, label='Raw Signal')
    plt.plot(x_fit * 1e9, functional_function(x_fit, *fit_params) * 1e3, linestyle='-', color='blue', linewidth=2, label='Functional Fit')
    plt.plot(x_fit * 1e9, thermal_function(x_fit, *fit_params) * 1e3, linestyle='-', color='red', linewidth=2, label='Thermal Fit')

    plt.xlabel('Time [ns]', fontsize=16, labelpad=10)
    plt.ylabel('Heterodyne Diode Response [mV]', fontsize=16, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.75)
    plt.legend(fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    save_dir = Path('figures') / 'tgs'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'tgs-{idx:04d}.png'
    plt.savefig(save_path, dpi=600)

def plot_fft_lorentzian(fft, lorentzian_function, popt, file_idx):
    frequencies, amplitudes = fft[:, 0], fft[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies * 1e-6, amplitudes, linestyle='-', color='black', linewidth=2, label='FFT Signal')
    
    x_smooth = np.linspace(min(frequencies), max(frequencies), 1000)
    y_fit = lorentzian_function(x_smooth, *popt)
    plt.plot(x_smooth * 1e-6, y_fit, linestyle='-', color='red', linewidth=2, label='Lorentzian Fit')
    
    plt.xlabel('Frequency [MHz]', fontsize=16, labelpad=10)
    plt.ylabel('Normalized Amplitude', fontsize=16, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.75)
    plt.legend(fontsize=16)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    save_dir = Path('figures') / 'fft-lorentzian'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'fft-lorentzian-{file_idx:04d}.png'
    plt.savefig(save_path, dpi=600)

def plot_processed(signal, max_time, start_time, file_idx):
    time, amplitude = signal[:NUM_POINTS, 0], signal[:NUM_POINTS, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(time * 1e9, amplitude * 1e3, linestyle='-', color='black', linewidth=2, label='Processed Signal')
    
    y_range = plt.ylim()
    plt.vlines(max_time * 1e9, y_range[0], y_range[1], color='blue', linestyle='--', linewidth=2, label='Max Time')
    plt.vlines(start_time * 1e9, y_range[0], y_range[1], color='red', linestyle='--', linewidth=2, label='Start Time')
    
    plt.xlabel('Time [ns]', fontsize=16, labelpad=10)
    plt.ylabel('Heterodyne Diode Response [mV]', fontsize=16, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.75)
    plt.legend(fontsize=16)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    save_dir = Path('figures') / 'processed'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'processed-{file_idx:04d}.png'
    plt.savefig(save_path, dpi=600)

def create_app(signal_data, fit):
    app = dash.Dash(__name__)
    num_plots = len(signal_data)

    app.layout = html.Div([
        dcc.Graph(id='signal-plot', style={'height': '600px'}),
        html.Div([
            html.Button('❮', id='prev-button', n_clicks=0, style={'fontSize': '18px', 'margin': '0 10px'}),
            html.Button('❯', id='next-button', n_clicks=0, style={'fontSize': '18px', 'margin': '0 10px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '20px'}),
        html.Div(id='signal-indicator', style={'textAlign': 'center', 'fontSize': '18px'})
    ])

    @app.callback(
        [Output('signal-plot', 'figure'),
         Output('signal-indicator', 'children')],
        [Input('prev-button', 'n_clicks'),
         Input('next-button', 'n_clicks')],
        [State('signal-indicator', 'children')]
    )
    def update_plot(prev_clicks, next_clicks, current_signal):
        ctx = dash.callback_context
        if not ctx.triggered:
            idx = 0
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            current_idx = int(current_signal.split()[2]) - 1 if current_signal else 0
            idx = max(0, min(num_plots - 1, current_idx + (1 if button_id == 'next-button' else -1)))

        signal = np.array(signal_data[idx])
        param_keys = ['A[Wm^-2]', 'B[Wm^-2]', 'C[Wm^-2]', 'alpha[m^2s^-1]', 'beta[s^0.5]', 'theta[]', 'tau[s]', 'f[Hz]']

        fit_dict = fit.iloc[idx].to_dict()
        fit_params = [float(fit_dict[key]) for key in param_keys]
        start_idx = fit_dict['start_idx']

        start_time, grating = fit_dict['start_time'], fit_dict['grating_value[um]']
        functional_function, thermal_function = tgs_function(start_time, grating)
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(go.Scatter(
            x=signal[:NUM_POINTS, 0], 
            y=signal[:NUM_POINTS, 1], 
            mode='lines', 
            name='Raw Signal'
        ))

        x_fit = signal[start_idx:NUM_POINTS, 0]
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=functional_function(x_fit, *fit_params),
            mode='lines',
            name='Functional Fit'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=thermal_function(x_fit, *fit_params),
            mode='lines',
            name='Thermal Fit'
        ))

        fig.update_layout(
            title=f'TGS Signal {idx + 1}',
            xaxis_title='Time [s]',
            yaxis_title='Amplitude [V]',
            legend_title='Fit Type',
            height=600
        )

        return fig, f'Viewing Signal {idx + 1} of {num_plots}'

    return app

def plot_interactive(signal_data, fit):
    app = create_app(signal_data, fit) 
    app.run_server(debug=True)
