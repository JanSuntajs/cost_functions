"""
This module implements the routines
for calculation of the cost function.


"""
import numpy as np
from inspect import signature, getmembers, isfunction

from . import crit_functions
from . import rescale_functions

# --------------------------
# CRITICAL PARAMETER SCALING
# --------------------------

# which routines are available
_crit_functions_list = [fun for fun in getmembers(crit_functions)
                        if isfunction(fun[1])]
# dictionary to be used
_crit_functions_dict = {key.split('_x_crit_')[1]: value for
                        (key, value) in _crit_functions_list}
_crit_keys = _crit_functions_dict.keys()

# -------------------
# RESCALING FUNCTIONS
# -------------------
_resc_functions_list = [fun for fun in getmembers(rescale_functions)
                        if isfunction(fun[1])]

_resc_functions_dict = {key.split('_rescale_')[1]: value for
                        (key, value) in _resc_functions_list}
_resc_keys = _resc_functions_dict.keys()


# which routines are available


def pad_vals(x, y):
    """
    A helper routine ensuring all the
    input arrays are of the same shape.

    """

    xshapes = []
    for x_arr in x:

        x_arr = np.array(x_arr)
        xshapes.append(x_arr.shape[0])

    x_max_shape = np.max(xshapes)
    for i, x_arr in enumerate(x):

        pad_len = x_max_shape - x_arr.shape[0]
        x[i] = np.pad(x_arr, (0, pad_len), mode='constant',
                      constant_values=np.NaN)
        y[i] = np.pad(y[i], (0, pad_len), mode='constant',
                      constant_values=np.NaN)

    x = np.vstack(x).astype(np.float64)
    y = np.vstack(y).astype(np.float64)
    return x, y


def rescale_xvals(x, sizelist, x_crit_fun, rescale_fun,
                  rescale_type,
                  *args):
    """
    Rescale the independent variable values by:

    x_ = size / np.exp(a / (x - x_crit) ** 0.5)

    Where x_crit is given as:

    x_crit = x_crit_fun(size, *params)

    Parameters:
    -----------

    x: 2D array or 2D array-like
    An array of independent variable values

    sizelist: 1D array or 1D array-like
    A list of system size in consideration

    x_crit_fun: string
    String specifying a key which invokes
    an appropriate function to be used for
    finding the scaling of a critical
    parameter. The function should have the
    following interface:
    x_crit_fun(sizelist, *params_crit)

    rescale_fun: string
    String specifying a rescaling function
    to be invoked.
    Function should have the following
    interface:
    rescale_fun(size, *params_rescale)

    rescale_type: string
    String specifying whether rescaling
    takes place throug subtracting critical
    parameter values or through dividing
    by them. In other words:
    if rescale_type = 'sub' (subtract)
    x_ = x - x_crit
    if rescale_type = 'div' (divide)
    x_ = x / x_crit

    *params_crit, *params_rescale: tuple
    Parameters for the x_crit_fun and
    rescale_fun, respectively.

    Returns:
    --------

    rescale_x: 2D ndarray
    A 2D array of rescaled independent
    variable values for each involved
    system size.

    Raises:
    -------
    ValueError if x_crit_fun or rescale_fun
    strings would attempt to invoke functions
    that have not yet been defined in
    the crit_functions and rescale_functions
    modules, respectively.

    """

    if x_crit_fun not in _crit_keys:

        err_message = ('Critical parameter scaling '
                       'function {} not allowed! '
                       'Allowed functions are: {}').format(x_crit_fun,
                                                           _crit_keys)
        raise ValueError(err_message)

    if rescale_fun not in _resc_keys:
        err_message = ('Rescaling '
                       'function {} not allowed! '
                       'Allowed functions are: {}').format(rescale_fun,
                                                           _resc_keys)
        raise ValueError(err_message)

    rescale_fun = _resc_functions_dict[rescale_fun]
    x_crit_fun = _crit_functions_dict[x_crit_fun]
    # number of model's free parameters
    sig = signature(rescale_fun)

    # two mandatory arguments for the rescale
    # function are the x-values and the list of
    # system sizes; all other arguments are relevant
    # for the critical point scaling
    npar = len(sig.parameters) - 2
    npar_ = len(args) - npar
    params_crit = args[:npar_]
    params_rescale = args[npar_:]

    x_crit = x_crit_fun(sizelist, *params_crit)

    # whether to subtract the "critical" values
    # or divide by them
    if rescale_type == 'sub':
        x_ = x - x_crit[:, np.newaxis]
    elif rescale_type == 'div':
        x_ = x / x_crit[:, np.newaxis]
    else:
        raise ValueError('rescale_x info: rescale_type not given!')

    rescale_x = rescale_fun(x_, sizelist, *params_rescale)

    return rescale_x


def minimization_fun(params, x, y, sizelist, x_crit_fun,
                     rescale_fun, rescale_type, return_data=False):
    """
    A minimization function for which the cost function
    should be minimized in order to obtain the optimum
    collapse of the data for a chosen functional
    dependence of the examined data.

    Parameters:
    -----------

    params: list
    A list of floats representing the parameters of the
    fitting/rescaling function to be used in the procedure.
    The values of params are the ones to be sought after
    by the minimization procedure.

    x: 2D array or 2D array-like
    A 2D list of independent variable values for different
    system sizes.

    y: 2D list or 2D array-like
    A 2D list of dependent variable values for different system
    sizes.

    sizelist: 1D array or 1D array-like
    A 1D list of system sizes used.


    """
    #x, y = _pad_vals(x, y)

    x_rescaled = rescale_xvals(x, sizelist, x_crit_fun, rescale_fun,
                               rescale_type,
                               *params)

    x_min = x_rescaled.flatten()
    y_min = y.flatten()

    sort_args = np.argsort(x_min)
    y_min = y_min[sort_args]

    # define the cost function
    cost_fun = np.nansum(np.abs(np.diff(y_min))) / \
        (np.nanmax(y_min) - np.nanmin(y_min)) - 1

    if not return_data:

        return cost_fun
    else:
        return np.array(x_min)[sort_args], y_min, cost_fun
