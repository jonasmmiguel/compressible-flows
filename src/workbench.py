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


def calculate_mach(input, mode='nshock', maxiter=500):
    """
    Calculates M resulting in the input.
    In general, there is no explicit relation for Mach number as a function of known ratios (e.g. T/Tt).
    We retrieve the most plausible value for M via an optimization process.

    :param input:
    :param mode:
    :param maxiter:
    :return: M resulting in the input
    """
    k = get_k(input)

    try:
        known_ratio_type = list(set(input.keys()) - set(['k', 'regime']))[0]  # e.g. 'Tt'
        try:
            known_ratio_value = input[known_ratio_type].magnitude
        except AttributeError:
            known_ratio_value = input[known_ratio_type]

        # set default optimizer
        optimizer = 'brent'

        # define the loss function, i.e. the function we want to minimize
        if mode == 'nshock':
            input['regime'] = 'supersonic'
            loss = lambda Ms, k: (nshock(known_ratio_type, Ms=Ms, k=k) - known_ratio_value) ** 2  # squared error
        elif mode == 'isentropic':
            loss = lambda M, k: (isentropic(known_ratio_type, M=M, k=k) - known_ratio_value) ** 2
        elif mode == 'fanno':
            loss = lambda M, k: (fanno(known_ratio_type, M=M, k=k) - known_ratio_value) ** 2
        elif mode == 'rayleigh':
            loss = lambda M, k: (rayleigh(known_ratio_type, M=M, k=k) - known_ratio_value) ** 2
            if (known_ratio_type == 'T') and (input['T'] > 1.0):
                optimizer = 'bruteforce'
        else:
            NotImplementedError('Invalid flow function family {}.'.format(mode))

        M = find_minimum(loss, input, k, maxiter=maxiter, optimizer=optimizer)
        return M

    except KeyError:
        NotImplementedError(
            'Invalid input type ({}).'.format(input))


def find_minimum(loss, input, k, maxiter=500, optimizer='brent'):
    if input['regime'] == 'subsonic':
        M_range = [0, 1 - 1E-08]
    elif input['regime'] == 'supersonic':
        M_range = [1 + 1E-08, 10]

    if optimizer == 'brent':
        M = minimize_scalar(fun=loss,
                            args=(k),
                            method='bounded',  # bounded Brent optimizer
                            bounds=M_range,
                            options={'xatol': 1E-08,
                                     'maxiter': maxiter,
                                     'disp': True, }
                            )['x']

    elif optimizer == 'bruteforce':
        M = brute(func=loss,
                  args=(k,),
                  ranges=(slice(0.70, 1.00, 1E-03),),
                  disp=True,
                  finish=None,
                  )
    return M


def isentropic(output_type, **input):
    k = get_k(input)

    if output_type == 'M':
        M = calculate_mach(input, mode='isentropic', maxiter=1000)
        return M
    else:
        M = input['M']
        if output_type == 'T':
            return 1 / (1 + 0.5 * (k - 1) * M ** 2)

        elif output_type == 'p':
            return isentropic('T', M=M, k=k) ** (k / (k - 1))

        elif output_type == 'A':
            return (1 / M) * ((1 + (0.5 * (k - 1)) * M ** 2) / (0.5 * (k + 1))) ** (
                    0.5 * (k + 1) / (k - 1))


def nshock(output_type, **input):
    k = get_k(input)

    if output_type == 'Ms':
        Ms = calculate_mach(input, mode='nshock', maxiter=1500)
        return Ms
    else:
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


def fanno(output_type, **input):
    k = get_k(input)

    if output_type == 'M':
        M = calculate_mach(input, mode='fanno', maxiter=1000)
        return M
    else:
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


def rayleigh(output_type, **input):
    k = get_k(input)

    if output_type == 'M':
        M = calculate_mach(input, mode='rayleigh', maxiter=1000)
        return M
    else:
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
    Ms_ray1 = rayleigh('M', T=0.6, regime='subsonic')
    Ms_ray2 = rayleigh('M', T=0.6, regime='supersonic')
    print('done')
