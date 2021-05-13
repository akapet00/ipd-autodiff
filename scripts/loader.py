import os

import pandas as pd
from scipy.io import loadmat


SUPPORTED_FREQS = [3., 3.5, 6., 10., 15., 20., 30., 40., 60., 80., 100.]


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
    pandas.DataFrame
        current distribution over the wire alongside additional data
    """
    assert frequency / 1e9 in SUPPORTED_FREQS, \
        (f'{frequency / 1e9} is not in supported. '
         f'Supported frequency values: {SUPPORTED_FREQS}.')
    data_dir = loadmat(os.path.join('data', 'fs_current', 'dataset.mat'))
    df = pd.DataFrame(data_dir['output'],
                      columns=['N', 'f', 'L', 'V', 'x', 'ireal', 'iimag'])
    df_f = df[df.f == frequency]
    df_f.reset_index(drop=True, inplace=True)
    return df_f
