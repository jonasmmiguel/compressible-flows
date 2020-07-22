# Author: Jonas M. Miguel (jonasmmiguel@gmail.com)

import numpy as np
from scipy.optimize import minimize, minimize_scalar, brute, fmin  # optimization algorithms
import matplotlib.pyplot as plt


def get_k(input):
    try:
        return input['k']
    except KeyError:
        return 1.4
    else:
        ValueError('k is undefined')


def find_minimum(loss, input, k, maxiter=500, method='brent'):
    if input['regime'] == 'subsonic':
        M_range = [0, 1 - 1E-08]
    elif input['regime'] == 'supersonic':
        M_range = [1 + 1E-08, 10]

    if method == 'brent':
        M = minimize_scalar(fun=loss,
                            args=(k),
                            method='bounded',
                            bounds=M_range,
                            options={'xatol': 1E-08,
                                     'maxiter': maxiter,
                                     'disp': True, }
                            )['x']
    elif method == 'bruteforce':
        M = brute(func=loss,
                  args=(k,),
                  ranges=(slice(0.70, 1.00, 1E-03),),
                  disp=True,
                  finish=None,
                  )
    return M


def isentropic(output_type, **input):
    k = get_k(input)

    try:
        M = input['M']
        if output_type == 'T':
            return 1 / (1 + 0.5 * (k - 1) * M ** 2)

        elif output_type == 'p':
            return isentropic('T', M=M, k=k) ** (k / (k - 1))

        elif output_type == 'A':
            return (1 / M) * ((1 + (0.5 * (k - 1)) * M ** 2) / (0.5 * (k + 1))) ** (
                    0.5 * (k + 1) / (k - 1))

    except KeyError:
        try:
            known_ratio = list(set(input.keys()) - set(['k', 'regime']))[0]  # The e.g. 'Tt'
            loss = lambda M, k: (isentropic(known_ratio, M=M, k=k) - input[
                known_ratio]) ** 2  # function we want to minimize. We use the squared error for penalizing (1) negative and positive deviations equaly and (2) w/ high smoothness
            M = find_minimum(loss, input, k, maxiter=1000)
            return M
        except KeyError:
            return KeyError('You probably forgot to specify the Mach regime. ' \
                            'Cannot determine M without pre-specifying the Mach regime.')

    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))


def nshock(output_type, **input):
    k = get_k(input)

    try:
        Ms = input['Ms']
        if output_type == 'Msl':
            term1 = Ms ** 2 + 2 / (k - 1)
            term2 = (2 * k / (k - 1)) * Ms ** 2 - 1
            return (term1 / term2) ** (1 / 2)

        elif output_type == 'pt':
            term1a = ((k + 1) / 2) * Ms ** 2
            term1b = 1 + ((k - 1) / 2) * Ms ** 2
            term3 = (2 * k / (k + 1)) * Ms ** 2 - ((k - 1) / (k + 1))
            return (term1a / term1b) ** (k / (k - 1)) * term3 ** (1 / (1 - k))

        elif output_type == 'p':
            Msl = nshock('Msl', Ms=Ms)
            return (1 + k * Ms ** 2) / (1 + k * Msl ** 2)

    except KeyError:
        try:
            known_ratio = list(set(input.keys()) - set(['k']))[0]  # e.g. 'Tt'
            loss = lambda Ms, k: (nshock(known_ratio, Ms=Ms, k=k) - input[known_ratio]) ** 2  # squared error
            input['regime'] = 'supersonic'
            Ms = find_minimum(loss, input, k, maxiter=1500)
            return Ms
        except KeyError:
            NotImplementedError(
                'Something went wrong. Check if input ({}), output ({}) configuration makes sense.'.format(input,
                                                                                                           output_type))
    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))


def fanno(output_type, **input):
    k = get_k(input)

    try:
        M = input['M']
        if output_type == 'T':
            return ((k + 1) / 2) * isentropic('T', M=M, k=k)

        elif output_type == 'p':
            T_ratio = fanno('T', M=M, k=k)
            return (1 / M) * T_ratio ** 0.5

        elif output_type == 'pt':
            T_ratio = fanno('T', M=M, k=k)
            return (1 / M) * T_ratio ** (-(k + 1) / (2 * (k - 1)))

        elif output_type == 'fld':
            T_ratio = fanno('T', M=M, k=k)
            term1 = ((k + 1) / (2 * k)) * np.log(T_ratio * M ** 2)
            term2 = (1 / k) * ((1 / M ** 2) - 1)
            return term1 + term2

        elif output_type == 'dels':
            return np.log((1 / M) * ((1 + 0.5 * (k - 1) * M ** 2) / (1 + 0.5 * (k - 1))) ** ((k + 1) / (2 * (k - 1))))

    except KeyError:
        try:
            known_ratio = list(set(input.keys()) - set(['k', 'regime']))[0]  # The e.g. 'Tt'
            loss = lambda M, k: (fanno(known_ratio, M=M, k=k) - input[
                known_ratio]) ** 2  # function we want to minimize. We use the squared error for penalizing (1) negative and positive deviations equaly and (2) w/ high smoothness
            M = find_minimum(loss, input, k, maxiter=1000)
            return M
        except KeyError:
            return KeyError('You probably forgot to specify the Mach regime. ' \
                            'Cannot determine M without pre-specifying the Mach regime.')
    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))


def rayleigh(output_type, **input):
    k = get_k(input)

    try:
        M = input['M']
        if output_type == 'Tt':
            return (2 * ((1 + k) * M ** 2) / (1 + k * M ** 2) ** 2) * (1 + ((k - 1) / 2) * M ** 2)
        elif output_type == 'T':
            return ((M * (1 + k)) ** 2) / ((1 + k * M ** 2) ** 2)
        elif output_type == 'p':
            return (1 + k) / (1 + k * M ** 2)
        elif output_type == 'pt':
            return rayleigh('p', M=M, k=k) * \
                   ((1 + ((k - 1) / 2) * M ** 2) / ((k + 1) / 2)) ** (k / (k - 1))
        elif output_type == 'dels':
            return (k / (k - 1)) * np.log(1 / (M ** 2)) + ((k + 1) / (k - 1)) * np.log((1 + k * M ** 2) / (1 + k))
    except KeyError:
        try:
            known_ratio = list(set(input.keys()) - set(['k', 'regime']))[0]  # e.g. 'Tt'
            loss = lambda M, k: (rayleigh(known_ratio, M=M, k=k) - input[known_ratio]) ** 2  # squared error
            if (known_ratio == 'T') and (input['T'] > 1.0):
                method = 'bruteforce'
            else:
                method = 'brent'
            M = find_minimum(loss, input, k, maxiter=1000, method=method)
            return M
        except KeyError:
            return KeyError('You probably forgot to specify the Mach regime. ' \
                            'Cannot determine M without pre-specifying the Mach regime.')
    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))


def colebrook(eps_to_D, Re):
    loss = lambda f: ((1 / f) ** 0.5 + (
                2 * np.log10((eps_to_D / 3.7) + (2.51 / (Re * (f ** 0.5)))))) ** 2  # squared error

    f0 = 0.05
    f = minimize(loss, x0=f0, bounds=[(0.008, 0.10)], method='TNC',
                 options={'maxiter': 10000, 'ftol': 1e-8})['x'][0]
    return f


if __name__ == '__main__':
    # eps_to_D = 0.01
    # Re = 3E+04
    # f = colebrook(eps_to_D, Re)

    # M = isentropic('M', A=3.27793, regime='subsonic')
    # Msl = nshock('Msl', Ms=1.5)
    #
    Ms2 = nshock('Ms', Msl=0.7)
    print('done')
