"""
This module implements
functions for predicting
the scaling behaviour
of the critical point.

The functions provided here
are meant to be used in
the rescale_xvals and
minimization_fun
functions from the costfun
module.

The function names should conform
to the following naming convention:
_x_crit_<some_descriptive_suffix>



"""
import numpy as np


def _x_crit_free(sizelist, *params):
    """
    Determine the critical parameter
    for each system size in question
    independently.

    Parameters:
    -----------

    sizelist: 1D array or 1D array-like
    Array of system sizes involved in
    the scaling analysis.

    params: float
    Variable number of the size-dependent
    critical parameter values to be
    determined. The number of params
    should match the number of system
    sizes involved.

    Returns:
    --------

    params: 1D ndarray
    The output of this function
    matches the input.
    """

    return np.array(params)


def _x_crit_poly(sizelist, *params):
    """
    Determine the critical parameter
    for each system size in question
    independently.

    Parameters:
    -----------

    sizelist: 1D array or 1D array-like
    Array of system sizes involved in
    the scaling analysis.

    params: float
    Variable number of the size-dependent
    critical parameter values to be
    determined. The number of params
    should match the number of system
    sizes involved.

    Returns:
    --------

    params: 1D ndarray
    The output of this function
    matches the input.
    """

    return np.array(params)


def _x_crit_lin(sizelist, x0, x1):
    """
    A model for linear scaling of
    the critical disorder strength
    with system size according to the
    equation:

    x_crit = x0 + x1 * size

    Parameters:
    -----------

    sizelist: 1D array or 1D array-like
    Array of system sizes involved in
    the scaling analysis.

    x0: float
    The subleading term in the linear
    dependence.

    x1: float
    The leading term in the linear dependence.

    Returns:
    --------

    1D ndarray:
    An array of critical parameter values
    for each involved system size.
    """

    return x0 + np.array(sizelist) * x1


def _x_crit_inv(sizelist, x0, x1):
    """
    A model for inverse scaling of
    the critical disorder strength
    with system size according to the
    equation:

    x_crit = x0 + x1 / size

    Parameters:
    -----------

    sizelist: 1D array or 1D array-like
    Array of system sizes involved in
    the scaling analysis.

    x0: float
    The subleading term in the inverse
    dependence.

    x1: float
    The leading term in the inverse dependence.

    Returns:
    --------

    1D ndarray:
    An array of critical parameter values
    for each involved system size.
    """

    x0, x1 = list(map(np.float64, [x0, x1]))
    return x0 + x1 / np.array(sizelist)


def _x_crit_inv_log(sizelist, x0, x1):
    """
    A model for inverse log scaling of
    the critical disorder strength
    with system size according to the
    equation:

    x_crit = x0 + x1 / log(size)

    Parameters:
    -----------

    sizelist: 1D array or 1D array-like
    Array of system sizes involved in
    the scaling analysis.

    x0: float
    The subleading term in the proposed
    dependence.

    x1: float
    The leading term in the proposed dependence.

    Returns:
    --------

    1D ndarray:
    An array of critical parameter values
    for each involved system size.
    """
    x0, x1 = list(map(np.float64, [x0, x1]))
    return x0 + x1 / np.log(sizelist)


