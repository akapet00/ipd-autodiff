import pandas as pd
from scipy.io import loadmat


SUPPORTED_FREQS = [3., 6., 10., 15., 20., 30., 40., 60., 80., 100.]


def load_antenna_el_properties(frequency):
    r"""Return the current distribution over the thin wire half-dipole
    antenna. The data are obtained by solving the Pocklington integro-
    differential equation by using the indirect-boundary element
    method.

    Ref: Poljak, D. Advanced modeling in computational electromagnetic
    compatibility, Wiley-Interscience; 1st edition (March 16, 2007)

    Parameters
    ----------
    frequency : float
        operating frequency in GHz

    Returns
    -------
    numpy.ndarray
        current distribution over the wire
    """
    assert frequency / 1e9 in SUPPORTED_FREQS, \
        (f'{frequency / 1e9} is not in supported. '
         f'Supported frequency values: {SUPPORTED_FREQS}.')
    data = loadmat('current.mat')['output']
    df = pd.DataFrame(data,
                      columns=['L', 'N', 'r', 'f', 'x', 'ireal', 'iimag'])
    df_f = df[df.f == frequency]
    df_f.reset_index(drop=True, inplace=True)
    return df_f
