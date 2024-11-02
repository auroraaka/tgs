import matplotlib
import numpy as np

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt

from tgs import tgs_function

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']


def plot_time_vs_response(fit_data, x_raw, y_raw, x_fit, y_fit, idx):
    
    idx = idx - 1
    param_keys = ['A[Wm^-2]', 'B[Wm^-2]', 'C[Wm^-2]', 'alpha[m^2s^-1]', 'beta[s^0.5]', 'theta', 'tau[s]', 'SAW_freq[Hz]']
    fit_dict = fit_data.iloc[idx].to_dict()
    fit_params = [float(fit_dict[key]) for key in param_keys]

    start_time, grating = fit_dict['start_time'], fit_dict['grating_value[um]']
    functional, thermal = tgs_function(start_time, grating)
    x_raw, y_raw, x_fit, y_fit = np.array(x_raw[idx]), np.array(y_raw[idx]), np.array(x_fit[idx]), np.array(y_fit[idx])

    plt.figure(figsize=(10, 6))
    plt.plot(x_raw[:1000] * 1e9, y_raw[:1000] * 1e3, linestyle='-', color='black', linewidth=2, label='Raw Signal')
    plt.plot(x_fit[:1000] * 1e9, functional(x_fit[:1000], *fit_params) * 1e3, linestyle='-', color='blue', linewidth=2, label='Functional Fit')
    plt.plot(x_fit[:1000] * 1e9, thermal(x_fit[:1000], *fit_params) * 1e3, linestyle='-', color='red', linewidth=2, label='Thermal Fit')

    plt.xlabel('Time [ns]', fontsize=16, labelpad=10)
    plt.ylabel('Heterodyne Diode Response [mV]', fontsize=16, labelpad=10)
    plt.xlim(0, 20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.75)
    plt.legend(fontsize=16)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('figures/time_vs_response.png', dpi=600)

def create_app(postfit_data, prefit_data, x_raw, y_raw, x_fit, y_fit):
    app = dash.Dash(__name__)
    num_plots = len(x_raw)

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

        xr, yr, xf, yf = np.array(x_raw[idx]), np.array(y_raw[idx]), np.array(x_fit[idx]), np.array(y_fit[idx])
        param_keys = ['A[Wm^-2]', 'B[Wm^-2]', 'C[Wm^-2]', 'alpha[m^2s^-1]', 'beta[s^0.5]', 'theta', 'tau[s]', 'SAW_freq[Hz]']

        prefit_dict = prefit_data.iloc[idx].to_dict()
        postfit_dict = postfit_data.iloc[idx].to_dict()
        prefit_params = [float(prefit_dict[key]) for key in param_keys]
        postfit_params = [float(postfit_dict[key]) for key in param_keys]

        start_time, grating = prefit_dict['start_time'], prefit_dict['grating_value[um]']
        functional, thermal = tgs_function(start_time, grating)

        fig = make_subplots(rows=1, cols=1)
        for name, data in [
            ('Raw Signal', yf),
            ('Functional Prefit', functional(xf, *prefit_params)),
            ('Thermal Prefit', thermal(xf, *prefit_params)),
            ('Functional Postfit', functional(xf, *postfit_params)),
            ('Thermal Postfit', thermal(xf, *postfit_params)),
        ]:
            fig.add_trace(go.Scatter(x=xf[:1000], y=data[:1000], mode='lines', name=name))

        fig.update_layout(
            title=f'TGS Signal {idx + 1}',
            xaxis_title='Time',
            yaxis_title='Amplitude',
            legend_title='Fit Type',
            height=600
        )

        return fig, f'Viewing Signal {idx + 1} of {num_plots}'

    return app

def plot_interactive(postfit, prefit, x_raw, y_raw, x_fit, y_fit):
    app = create_app(postfit, prefit, x_raw, y_raw, x_fit, y_fit) 
    app.run_server(debug=True)