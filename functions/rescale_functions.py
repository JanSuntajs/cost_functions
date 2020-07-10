"""
This module implements
functions for properly
rescaling the independent
variable data.

All the functions specified
in this module should have
the following interface:

rescale_fun(x, sizelist, *params)

Parameters:
-----------

x: 1D ndarray or 1D ndarray-like
Independent variable data to be
rescaled.

sizelist: 1D ndarray or 1D ndarray-like
List of sizes involved in the scaling
analysis.

*params: additional model (rescaling function)
parameters.

The functions' names should conform to the following
scheme:
_rescale_<some_descriptive_suffix>
"""
import numpy as np


# rescaling functions

def _rescale_kt(x, sizelist, a):
    """
    Rescaling according to the
    Kosterlitz-Thouless (KT)
    scaling law.

    Rescaling function:

    x_KT = sgn(x) * size / exp(a / abs(x) ** 0.5)

    NOTE: values of x entering the above equation have
    already been modified by subtracting the value of
    the critical parameter x_c from them
    """

    sizelist = np.array(sizelist)
    rescale_x = np.sign(
        x) * sizelist[:, np.newaxis] / np.exp(a / np.abs(x) ** 0.5)

    return rescale_x


def _rescale_pl(x, sizelist, nu):
    """
    Rescaling according to the
    power-law (pl) scaling.

    Rescaling function:

    x_pl = sgn(x) * size * abs(x) ** nu

    NOTE: values of x entering the above equation have
    already been modified by subtracting the value of
    the critical parameter x_c from them
    """
    nu = np.float64(nu)
    sizelist = np.array(sizelist)

    rescale_x = np.sign(x) * sizelist[:, np.newaxis] * np.abs(x) ** nu

    return rescale_x


def _rescale_id(x, sizelist, *params):
    """
    Identity function for rescaling which
    simply leaves the incoming x-data
    the way they are.

    """

    return x
