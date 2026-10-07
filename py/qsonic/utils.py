"""Utility functions for qsonic."""
import argparse

import numpy as np


def float_range(f1, f2):
    """ Returns a function that checks if a float is within a range."""
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range.
        """
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < f1 or f > f2:
            raise argparse.ArgumentTypeError(f"must be in range [{f1}--{f2}]")
        return f

    # Return function handle to checking function
    return float_range_checker


def get_data_case_insensitive(data, colname, dtype='d'):
    """Returns a column from a named numpy array, ignoring case of the column
    name."""
    result = None
    for x in [colname, colname.upper(), colname.lower()]:
        if x in data.dtype.names:
            result = np.array(data[x], dtype=dtype)
            break
    else:
        raise KeyError(f"Failed to get {colname}.")
    return result


def get_data_full_fallback(data, expected_colnames):
    """Returns a column from a named numpy array, ignoring case of the column
    name. Tries all expected column names in order until one is found."""
    result = None
    for colname in expected_colnames:
        try:
            result = get_data_case_insensitive(data, colname)
            break
        except KeyError:
            continue
    else:
        raise KeyError(f"Failed to get any of {expected_colnames}.")

    return result


def get_lambda(data):
    """Returns the wavelength array from a named numpy array, checking for
    'LAMBDA', 'LOGLAM', or 'lambda' columns. The output is always in wavelength
    space."""
    if 'LAMBDA' in data.dtype.names:
        return data['LAMBDA']
    elif 'LOGLAM' in data.dtype.names:
        return 10**data['LOGLAM']
    else:
        return data['lambda']