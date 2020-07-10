"""
Contains routines for parsing
the command-line arguments.

"""

import argparse


def arg_parser(system_keys, module_keys):
    """
    Get the values of the commmand line
    arguments which are to be given to the
    main script.
    """

    parser = argparse.ArgumentParser(prog='Obtain runtime command arguments '
                                     'for the executable script.')

    spec_system_keys = ['pbc', 'disorder', 'ham_type', 'model']
    for key in system_keys:
        if key not in spec_system_keys:
            type_ = int
            default = 0
            parser.add_argument('--{}'.format(key),
                                type=type_, default=default)
        else:
            # pbc or obc (periodic or open boundary conditions)
            if key == 'pbc':
                parser.add_argument('--{}'.format(key),
                                    type=str2bool, default=True)
            # select the disorder type
            if key == 'disorder':
                parser.add_argument('--{}'.format(key),
                                    type=str, default='none')

            # select the hamiltonian type ->
            # can be spin1d, ferm1d, or free
            if key == 'ham_type':
                parser.add_argument('--{}'.format(key),
                                    type=str, default='spin1d')
            # select the actual physical model, such as
            # the heisenberg or imbrie model
            if key == 'model':
                parser.add_argument('--{}'.format(key),
                                    type=str, default='')

    for key in module_keys:

        if 'seed' in key:
            argtype = int
        else:
            argtype = float

        parser.add_argument('--{}'.format(key),
                            type=argtype, default=argtype(0.0))

    for name in ['--results', '--syspar', '--modpar']:
        parser.add_argument(name, type=str, default='.')

    args, extra = parser.parse_known_args()
    return vars(args), extra


def arg_parser_general(*args):
    """
    A general function for parsing the command-line arguments.

    Parameters
    ----------

    args: a tuple of dictionaries, each of the form:

    {'cmd_arg': [type_, default_value],}

    Here, 'cmd_arg' key specifies the command-line argument to be
    parsed and [type_, default_value] list specifies the type
    of the argument to be parsed and default_value specifies
    the default value which should be given if no value is parsed.


    """

    parser = argparse.ArgumentParser(prog='Obtain runtime command arguments '
                                     'for the executable script.')

    for arg in args:

        for key, value in arg.items():

            type_, default = value

            parser.add_argument('--{}'.format(key),
                                type=type_, default=default)

    args, extra = parser.parse_known_args()

    return vars(args), extra
