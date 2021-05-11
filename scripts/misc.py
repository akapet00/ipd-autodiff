import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


# visualization
def fig_config(latex=False, nrows=1, ncols=1, scaler=1.0):
    r"""Configure matplotlib parameters for better visualization style.
    
    Parameters
    ----------
    latex : bool, optional
        If true, LaTeX backend will be used
    nrows : int, optional
        number of figures row-wise
    ncols : int, optional
        number of figures column-wise
    scaler : float, optional
        scaler for each figure
        
    Returns
    -------
    None
    """
    plt.rcParams.update({
        'text.usetex': latex,
        'font.family': 'serif',
        'font.size': 14,
        'figure.figsize': (4.774 * scaler * ncols, 2.950 * scaler * nrows),
        'lines.linewidth': 3,
        'lines.dashed_pattern': (3, 5),
        'lines.markersize': 10,
        'lines.markeredgecolor': 'k',
        'lines.markeredgewidth': 0.5,
        'image.origin': 'lower',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'grid.linewidth': 0.5,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })


def fig_config_reset():
    r"""Recover matplotlib default parameters.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    plt.rcParams.update(plt.rcParamsDefault)
    
    
# error evaulation
def mse(true, pred):
    r"""Return mean square difference between two values.
    
    Parameters
    ----------
    true : float or numpy.ndarray
        True value(s)
    pred : float or numpy.ndarray
        Predicted or simulated value(s)
    
    Returns
    -------
    float
        Root mean square error value
    """
    if (not isinstance(true, (jnp.ndarray, np.ndarray, int, float)) or
            not isinstance(pred, (jnp.ndarray, np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.mean((true - pred) ** 2)


def rmse(true, pred):
    r"""Return root mean square difference between two values.
    
    Parameters
    ----------
    true : float or numpy.ndarray
        True value(s)
    pred : float or numpy.ndarray
        Predicted or simulated value(s)
    
    Returns
    -------
    float
        Root mean square error value
    """
    if (not isinstance(true, (jnp.ndarray, np.ndarray, int, float)) or
            not isinstance(pred, (jnp.ndarray, np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.sqrt(mse(true, pred))


def msle(true, pred):
    r"""Return mean square log difference between two values.
    
    Parameters
    ----------
    true : float or numpy.ndarray
        True value(s)
    pred : float or numpy.ndarray
        Predicted or simulated value(s)
    
    Returns
    -------
    float
        Mean square log error value
    """
    if (not isinstance(true, (jnp.ndarray, np.ndarray, int, float)) or
            not isinstance(pred, (jnp.ndarray, np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return mse(np.log1p(true), np.log1p(pred))


def mae(true, pred):
    r"""Return mean absolute difference between two values.
    
    Parameters
    ----------
    true : float or numpy.ndarray
        True value(s)
    pred : float or numpy.ndarray
        Predicted or simulated value(s)
    
    Returns
    -------
    float
        Mean absolute error value
    """
    if (not isinstance(true, (jnp.ndarray, np.ndarray, int, float)) or
            not isinstance(pred, (jnp.ndarray, np.ndarray, int, float))):
        raise ValueError('supported data types: numpy.ndarray, int, float')
    return np.mean(np.abs(true - pred))
