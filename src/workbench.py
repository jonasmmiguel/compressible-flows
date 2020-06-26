# Author: Jonas M. Miguel (jonasmmiguel@gmail.com)

import numpy as np
from scipy.optimize import minimize, brute


def get_k(input):
    try:
        return input['k']
    except KeyError:
        return 1.4
    else:
        ValueError('k is undefined')


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
        if input['A_ratio'] and output_type == 'M':
            loss = lambda M, k: (isentropic('A', M=M, k=k) - input['A_ratio']) ** 2  # squared error

            if input['regime'] == 'subsonic':
                M_range = [(0, 1)]
                M_initial_guess = 0.3
            elif input['regime'] == 'supersonic':
                M_range = [(1, 10)]
                M_initial_guess = 1.6

            M = minimize(loss, x0=np.array(M_initial_guess), bounds=M_range, method='TNC', args=k,
                         options={'maxiter': 100, 'ftol': 1e-8})['x'][0]
            return M
    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))


def nshock(output_type, **input):
    k = get_k(input)

    try:
        Ms = input['Ms']
        if output_type == 'M':
            term1 = Ms ** 2 + 2 / (k - 1)
            term2 = (2 * k / (k - 1)) * Ms ** 2 - 1
            return (term1 / term2) ** (1 / 2)

        elif output_type == 'pt':
            term1a = ((k + 1) / 2) * Ms ** 2
            term1b = 1 + ((k - 1) / 2) * Ms ** 2
            term3 = (2 * k / (k + 1)) * Ms ** 2 - ((k - 1) / (k + 1))
            return (term1a / term1b) ** (k / (k - 1)) * term3 ** (1 / (1 - k))

        elif output_type == 'p':
            Msl = nshock('M', Ms=Ms)
            return (1 + k * Ms ** 2) / (1 + k * Msl ** 2)

    except KeyError:
        assert input['Msl'] != None
        loss = lambda M, k: (nshock('M', Ms=M, k=k) - input['Msl']) ** 2  # squared error

        # Ms is certainly supersonic
        M_range = [(1.01, 10)]
        M_initial_guess = 1.02

        M = minimize(loss, x0=np.array(M_initial_guess), bounds=M_range, method='TNC', args=k,
                      options={'maxiter': 10000, 'ftol': 1e-8})['x'][0]
        return M

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

    except KeyError:
        if input['fld'] and input['regime'] and output_type == 'M':
            loss = lambda M, k: (fanno('fld', M=M, k=k) - input['fld']) ** 2  # squared error

            if input['regime'] == 'subsonic':
                M_range = [(0, 1)]
                M_initial_guess = 0.3
            elif input['regime'] == 'supersonic':
                M_range = [(1, 10)]
                M_initial_guess = 1.6

            M = minimize(loss, x0=np.array(M_initial_guess), bounds=M_range, method='TNC', args=k,
                         options={'maxiter': 100, 'ftol': 1e-8})['x'][0]
            return M

    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))


def find_minimum(loss, input, k):
    if input['regime'] == 'subsonic':
        M_range = [(0, 0.9999)]
        M_initial_guess = np.linspace(0.1, 0.99, 99)
    elif input['regime'] == 'supersonic':
        M_range = [(1.0001, 10)]
        M_initial_guess = np.linspace(1.05, 3.00, 19)

    Mlist = []
    for M0 in M_initial_guess:
        M = minimize(loss, x0=M0,  bounds=M_range, method='TNC', args=k,
                     options={'maxiter': 100, 'ftol': 1e-5})['x'].mean()
        Mlist.append(M)

    M_out = np.median(Mlist)

    if (M_out > 0.72) and (M_out < 0.85):
        ValueError('Estimated value of Mach is probably wrong. Try sth btw. 0.7 and 0.99')

    return M_out


def rayleigh(output_type, **input):
    k = get_k(input)

    try:
        M = input['M']
        if output_type == 'Tt':
            return (2 * ((1 + k) * M ** 2) / (1 + k * M ** 2) ** 2) * (1 + ((k - 1) / 2) * M ** 2)
        elif output_type == 'T':
            return (M * (1 + k)) ** 2 / (1 + k * M ** 2) ** 2
        elif output_type == 'p':
            return (1 + k) / (1 + k * M ** 2)
        elif output_type == 'pt':
            return rayleigh('p', M=M, k=k) * \
                   ((1 + ((k - 1) / 2) * M ** 2) / ((k + 1) / 2)) ** (k / (k - 1))
    except KeyError:
        try:
            known_ratio = list(set(input.keys()) - set(['M', 'k', 'regime']))[0]  # e.g. 'Tt'
            loss = lambda M, k: (rayleigh(known_ratio, M=M, k=k) - input[known_ratio]) ** 2  # squared error
            M = find_minimum(loss, input, k)
            return M
        except KeyError:
            return KeyError('You probably forgot to specify the Mach regime. ' \
                'Cannot determine M without pre-specifying the Mach regime.')
    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))





if __name__ == '__main__':
    print('done')
