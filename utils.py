import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.special import binom
 

# finite differentiation
def central_difference(func, axis='x', args=(), order=1, eps=1.e-4):
    """Return n-th order central numerical difference of a given
    time-independent function.
    
    If order is not given, it is assumed to be 1.
    
    Args
    ----
    func (callable) - function to derive w.r.t. a single variable
    axis (string, optional) - differentiation domain 
    args (tuple, optional) - additional arguments of a function
    order (int, optional) - numerical derivation order
    eps (float, optional) - numerical derivation precision
    
    Parameters
    ----------
    func : callable
        function to derive w.r.t. a single variable
    axis : string, optional
        differentiation domain 
    args : tuple, optional)
        additional arguments of a function
    order : int, optional
        numerical derivation order
    eps : float, optional
        numerical derivation precision
    
    Returns
    -------
    numpy.ndarray
        central difference of func
    """
    if axis not in ['x', 'y', 'z']:
        raise ValueError('`x`, `y` and `z` axis are supported.')
    if order not in [1, 2]:
        raise ValueError(f'Differentiation order {order} is not supported.')
    precision_low = 1.e-2
    precision_high = 1.e-9
    if eps > precision_low:
        raise ValueError(f'`eps` has to be larger than {precision_low}.')
    elif eps < precision_high:
        raise ValueError(f'`eps` has to be less than {precision_high}.')
    if axis == 'x':
        def f(x):
            if order == 1:
                return (func(x+eps, *args) 
                        - func(x-eps, *args))/(2*eps)
            if order == 2:
                return (func(x+eps, *args) 
                        - 2*func(x, *args) 
                        + func(x-eps, *args))/eps**2
    elif axis == 'y':
        def f(y):
            if order == 1:
                return (func(args[0], y+eps, *args[1:]) 
                        - func(args[0], y-eps, *args[1:]))/(2*eps)
            if order == 2:
                return (func(args[0], y+eps, *args[1:]) 
                        - 2*func(args[0], y, *args[1:]) 
                        + func(args[0], y-eps, *args[1:]))/eps**2
    else:
        def f(z):
            if order == 1:
                return (func(*args[:2], z+eps, *args[2:]) 
                        - func(*args[:2], z-eps, *args[2:]))/(2*eps)
            if order == 2:
                return (func(*args[:2], z+eps, *args[2:]) 
                        - 2*func(*args[:2], z, *args[2:]) 
                        + func(*args[:2], z-eps, *args[2:]))/eps**2
    return f


def holoborodko(y, dx=1):
    """Return the 1st order numerical difference on a sampled data. If
    `dx` is not given, it is assumed to be 1. This function is to be
    used when noise is present in the data. Filter length of size 5 is
    used in this implementation. For more details check:
    http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
    
    Parameters
    ----------
    y : numpy.ndarray
        data to derive w.r.t. a single variable
    dx : float, optional
        elementwise distance
    
    Returns
    -------
    numpy.ndarray
        1st order numerical differentiation
    """
    N = 5
    M = (N-1) // 2
    m = (N - 3) // 2
    ck = [(1 / 2 ** (2 * m + 1) * (binom(2 * m, m - k + 1)
           - binom(2 * m, m - k - 1))) for k in range(1, M + 1)]
    if np.iscomplex(y).any():
        diff_type = 'complex_'
    else:
        diff_type = 'float'
    y_x = np.empty((y.size, ), dtype=diff_type)
    y_x[0] = (y[1] - y[0]) / dx
    y_x[1] = (y[2] - y[0]) / (2 * dx)
    y_x[-2] = (y[-1] - y[-3]) / (2 * dx)
    y_x[-1] = (y[-1] - y[-2]) / dx
    for i in range(M, len(y) - M):
        y_x[i] = 1 / dx * sum([ck[k - 1] * (y[i + k] - y[i - k]) for k
                               in range(1, M + 1)])
    return y_x


# numerical integration
def quad(func, a, b, args=(), n_points=3):
    """Return the the integral of a given function using the
    Gauss-Legendre quadrature scheme.
    
    Parameters
    ----------
    func : callable
        integrand
    a : float
        left boundary of the integration domain
    b : float
        right boundary of the integration domain
    args : tuple, optional
        additional arguments for `func`
    n_points : int, optional
        degree of the Gauss-Legendre quadrature
        
    Returns
    -------
    float
        integral of a given function
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    psi, w = np.polynomial.legendre.leggauss(n_points)
    xi = ((b - a) / 2) * psi + (a + b) / 2
    return (b - a) / 2 * w @ func(xi, *args)


def dblquad(func, bbox, args=(), n_points=9):
    """Return the the integral of a given 2-D function, `f(y, x)`,
    using the Gauss-Legendre quadrature scheme.
    
    Parameters
    ----------
    func : callable
        integrand
    a : list or tuple
        integration domain [min(x), max(x), min(y), max(y)]
    args : tuple, optional
        additional arguments for `func`
    n_points : int, optional
        degree of the Gauss-Legendre quadrature
        
    Returns
    -------
    float
        integral of a given function
    """
    if not callable(func):
        raise ValueError('`func` must be callable')
    psi, w = np.polynomial.legendre.leggauss(n_points)
    ay, by, ax, bx = bbox
    xix = (bx - ax) / 2 * psi + (ax + bx) / 2
    xiy = (by - ay) / 2 * psi + (ay + by) / 2
    return (bx - ax) / 2 * (by - ay) / 2 * w @ func(xiy, xix, *args) @ w


def elementwise_quad(y, x, n_points=3):
    """Return the approximate value of the integral of a given sampled
    data using the Gauss-Legendre quadrature.
    
    Parameters
    ----------
    y : numpy.ndarray
        sampled integrand
    x : numpy.ndarray
        integration domain
    n_points : int, optional
        degree of the Gauss-Legendre quadrature
        
    Returns
    -------
    float
        approximate of the integral of a given function
    """
    if not isinstance(y, (np.ndarray, jnp.ndarray)):
        raise Exception('`y` must be numpy.ndarray.')
    try:
        a = x[0]
        b = x[-1]
    except TypeError:
        print('`x` must be numpy.ndarray')
    func = interp1d(x, y, kind='cubic')
    return quad(func, a, b, n_points=n_points)


def elementwise_dblquad(z, x, y, n_points=9):
    """Return the approximate value of the integral of a given sampled
    2-D data using the Gauss-Legendre quadrature.
    
    Parameters
    ----------
    z: numpy.ndarray
        sampled integrand function of shape (x.size, y.size)
    y : numpy.ndarray
        y-axis strictly ascending coordinates
    x : numpy.ndarray
        x-axis strictly ascending coordinates
    n_points : int, optional
        degree of the Gauss-Legendre quadrature
        
    Returns
    -------
    float
        approximate of the integral of a given function
    """
    if not isinstance(y, (np.ndarray, jnp.ndarray)):
        raise Exception('`y` must be numpy.ndarray.')
    try:
        bbox = [y[0], y[-1], x[0], x[-1]]
    except TypeError:
        print('Both `x` and `y` must be numpy.ndarray')
    func = RectBivariateSpline(y, x, z, bbox=bbox)
    return dblquad(func, bbox, n_points=n_points)
