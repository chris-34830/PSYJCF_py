import logging
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..config import config

def _plot_vol_comparison(dfs, S, option_type, model_col, dates=None, use_moneyness=True):
    """
    Plot volatility smiles comparison across different maturities.
    
    Parameters:
    -----------
    dfs : list of pandas.DataFrame
        List of option chains for different maturities
    S : float
        Current stock price
    option_type : str
        Option type ('C' for call, 'P' for put)
    model_col : str
        Column containing implied volatilities to plot
    dates : list of str, optional
        List of expiration dates for labeling
    use_moneyness : bool
        If True, x-axis shows log(K/S), otherwise shows strike price
    """
    plt.figure(figsize=config['viz']['comparison_figsize'])

    for i, df in enumerate(dfs):
        # Filter valid data points
        df_plot = df.dropna(subset=["x_var", model_col]).copy()
        if df_plot.empty:
            continue

        # Create label with maturity and optional date
        label = f"T = {df_plot['T'].iloc[0]:.2f}"
        if dates and i < len(dates):
            label += f" ({dates[i]})"

        # Plot volatility curve
        xvals = df_plot["x_var"]
        yvals = df_plot[model_col]
        plt.plot(
            xvals,
            yvals,
            f'C{i}o--',  # Different color for each maturity
            label=label
        )

    # Customize plot
    xaxis_label = "log(K/S)" if use_moneyness else "Strike (K)"
    plt.title(f"Comparaison des Smiles - {option_type} - {model_col}")
    plt.xlabel(xaxis_label)
    plt.ylabel("Volatilité Implicite")
    plt.axvline(x=0 if use_moneyness else S, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

def _plot_single_model_surface(df, S, option_type, model_col, use_moneyness=True):
    """
    Create a 3D volatility surface plot for a single model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Combined option chain data for all maturities
    S : float
        Current stock price
    option_type : str
        Option type ('C' for call, 'P' for put)
    model_col : str
        Column containing implied volatilities to plot
    use_moneyness : bool
        If True, x-axis shows log(K/S), otherwise shows strike price
    """
    # Filter valid data points
    df_plot = df.dropna(subset=[model_col, "x_var", "T"]).copy()
    if df_plot.empty:
        logging.warning(f"Données insuffisantes pour la surface {option_type} - {model_col}")
        return
        
    # Create mesh grid for surface
    x_unique = np.sort(df_plot["x_var"].unique())
    times_unique = np.sort(df_plot["T"].unique())
    x_grid, time_grid = np.meshgrid(x_unique, times_unique)
    vol_matrix = np.zeros_like(x_grid)

    # Fill volatility matrix
    for i, t in enumerate(times_unique):
        for j, xval in enumerate(x_unique):
            mask = (df_plot["T"] == t) & (df_plot["x_var"] == xval)
            if any(mask):
                vol_matrix[i, j] = df_plot.loc[mask, model_col].iloc[0]

    # Create 3D surface plot
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_grid,
        y=time_grid,
        z=vol_matrix,
        colorscale='Viridis',
        opacity=0.7,
        showscale=True
    ))
    
    # Customize plot
    x_label = "log(K/S)" if use_moneyness else "Strike (K)"
    fig.update_layout(
        title=f'Surface de Volatilité 3D ({option_type}) - {model_col}',
        scene=dict(
            xaxis_title=x_label,
            yaxis_title='Temps à Maturité (années)',
            zaxis_title='Volatilité Implicite'
        ),
        width=config['viz']['surface_width'],
        height=config['viz']['surface_height']
    )
    
    fig.show()

def plot_volatility_surface(df, S, option_type, plot_type='surface', model_col=None, dates=None, **kwargs):
    """
    Main function to create volatility visualizations.
    
    Parameters:
    -----------
    df : pandas.DataFrame or list of DataFrame
        Option chain data. For 'comparison' plot_type, should be a list of DataFrames
    S : float
        Current stock price
    option_type : str
        Option type ('C' for call, 'P' for put)
    plot_type : str
        Type of plot to create ('surface' or 'comparison')
    model_col : str
        Column containing implied volatilities to plot
    dates : list of str, optional
        List of expiration dates for labeling in comparison plots
    **kwargs : 
        Additional plotting parameters
    """
    if plot_type == 'comparison':
        if not isinstance(df, (list, tuple)):
            logging.warning("Pour 'comparison', df doit être liste/tuple de DataFrames.")
            return
        if not model_col:
            logging.warning("Pas de 'model_col' pour la comparaison.")
            return
        _plot_vol_comparison(df, S, option_type, model_col, dates)

    elif plot_type == 'surface':
        if not model_col:
            logging.warning("Pas de 'model_col' pour la surface 3D.")
            return
        _plot_single_model_surface(df, S, option_type, model_col)

    else:
        logging.warning(f"Type de graphique non supporté: {plot_type}")
        return