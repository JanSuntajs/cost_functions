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
def _rescale_alt(x, sizelist, a, nu):
    """
    Rescaling according to the
    B. Altshuler's proposal, where an ansatz
    similar to the BKT scaling form is used
    but with an additional free parameter.

    Rescaling function:

    x_alt = sgn(x) * size / exp(a / abs(x) ** nu)

    NOTE: values of x entering the above equation have
    already been modified by subtracting the value of
    the critical parameter x_c from them
    """

    sizelist = np.array(sizelist)
    rescale_x = np.sign(
        x) * sizelist[:, np.newaxis] / np.exp(a / np.abs(x) ** nu)

    return rescale_x


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


def _rescale_pl_two_exponents(x, sizelist, nu1, nu2):
    """
    Rescaling according to the
    power-law (pl) scaling.

    Rescaling function:

    x_pl = sgn(x) * size * abs(x) ** nu;

    here, nu can be either nu1 or nu2, depending
    on whether we are fitting data on the right
    or left side of the critical point, respectively.

    NOTE: values of x entering the above equation have
    already been modified by subtracting the value of
    the critical parameter x_c from them
    """
    nu1, nu2 = list(map(np.float64, [nu1, nu2]))
    sizelist = np.array(sizelist)

    x_left = x[x <= 0]
    x_right = x[x > 0]

    x_left = np.sign(x_left) * sizelist[:, np.newaxis] * np.abs(x_left) ** nu1
    x_right = np.sign(x_right) * \
        sizelist[:, np.newaxis] * np.abs(x_right) ** nu2

    rescale_x = np.append(x_left, x_right)

    return rescale_x


def _rescale_pl_irrel(x, sizelist, nu, a0, a1, a2):
    """
    Rescaling according to the power-law scaling which includes
    the irrelevant contribution.
    Rescaling function:

    x_pl_irr = sgn(x) * size * abs(x) ** nu + a0*size**(-1)+
               a1*size + a2*size**2
    """

    rescale_x = _rescale_pl(x, sizelist, nu)

    sizelist = 1.0 * np.array(sizelist)
    sizelist = sizelist[:, np.newaxis]

    rescale_x += (a0 * sizelist**(-1.) + a1 * sizelist + a2 * sizelist**2.)

    return rescale_x


def _rescale_id(x, sizelist):
    """
    Identity function for rescaling which
    simply leaves the incoming x-data
    the way they are.

    """

    return x
