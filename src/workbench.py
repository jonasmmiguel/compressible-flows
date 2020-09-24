# Author: Jonas M. Miguel (jonasmmiguel@gmail.com)

import numpy as np
from scipy.optimize import root, minimize_scalar


def get_k(input):
    try:
        return input['k'].magnitude
    except AttributeError:
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


def find_minimum(loss, input, k, maxiter=500, optimizer='brent', display_optimizer_status=False):
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
                                     'disp': display_optimizer_status, }
                            )['x']

    elif optimizer == 'bruteforce':
        M = brute(func=loss,
                  args=(k,),
                  ranges=(slice(0.70, 1.00, 1E-03),),
                  disp=display_optimizer_status,
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


def BWR(Tr, vr, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, beta, gamma):
    """
    Calculate Z via the Benedict-Webb-Rubin Equation

    :param vr:
    :param Tr:
    :param b1:
    :param b2:
    :param b3:
    :param b4:
    :param c1:
    :param c2:
    :param c3:
    :param c4:
    :param d1:
    :param d2:
    :param beta:
    :param gamma:
    :return: Z: compressibility factor
    """
    B = b1 - (b2 / Tr) - (b3 / Tr ** 2) - (b4 / Tr ** 3)
    C = c1 - (c2 / Tr) + (c3 / Tr ** 3)
    D = d1 + (d2 / Tr)

    if abs(vr) < 1e-08:
        vr = 1e-08

    tail_term = (c4 / (Tr ** 3 * vr ** 2)) * (beta + gamma / vr ** 2) * np.exp(-gamma / vr ** 2)

    Z = 1 + (B / vr) + (C / vr ** 2) + (D / vr ** 5) + tail_term
    return Z


def acentric_factor(fluid):
    Tc, pc = fluid.critical()
    pref = fluid.ps( 0.7 * Tc )
    pref_r = pref / pc
    omega = - np.log(pref_r) / np.log(10) - 1
    return omega


def leekesler(Tr: float, vr: float, omega=0.0) -> float:
    """
    Coefficients according to
    A Generalized Thermodynamic Correlation Based on Three-Parameter Corresponding States
    Lee & Kesler 1975
    p. 511
    Table I
    """
    # simple fluid
    bwr_simplefluid_coeffs = {
        'b1': 0.1181193,
        'b2': 0.265728,
        'b3': 0.154790,
        'b4': 0.030323,
        'c1': 0.0236744,
        'c2': 0.0186984,
        'c3': 0.0,
        'c4': 0.042724,
        'd1': 0.155488E-04,  # 'd1': 0.155428E-04 (1978) OR 0.155488E-04 (VanWylen 8ed / Lee-Kesler 1975)
        'd2': 0.623689E-04,
        'beta': 0.65392,
        'gamma': 0.060167
    }

    # reference fluid
    omega_ref = 0.3978
    bwr_reffluid_coeffs = {
        'b1': 0.2026579,
        'b2': 0.331511,
        'b3': 0.027655,
        'b4': 0.203488,
        'c1': 0.0313385,
        'c2': 0.0503618,
        'c3': 0.016901,
        'c4': 0.041577,
        'd1': 0.48736E-04,
        'd2': 0.0740336E-04,
        'beta': 1.226,
        'gamma': 0.03754
    }

    bwe_inputs_simple = bwr_simplefluid_coeffs
    bwe_inputs_simple['Tr'] = Tr
    bwe_inputs_simple['vr'] = vr

    bwe_inputs_ref = bwr_reffluid_coeffs
    bwe_inputs_ref['Tr'] = Tr
    bwe_inputs_ref['vr'] = vr


    Z0 = BWR(**bwe_inputs_simple)

    if omega != 0:
        Zref = BWR(**bwe_inputs_ref)
        Z = Z0 + (omega / omega_ref) * (Zref - Z0)
    else:
        Z = Z0

    # if Z < 0:
    #     Z = 1
    return Z


def _fun(vr, Tr, pr):
    return (pr * vr) / Tr - leekesler(Tr=Tr, vr=vr)


def _loss(vr, Tr, pr):
    (((pr * vr) / Tr) - leekesler(Tr=Tr, vr=vr)) ** 2


def get_Z(pr: float, Tr: float) -> float:

    if type(pr) not in [float, int]:
        pr = pr.magnitude

    if type(Tr) not in [float, int]:
        Tr = Tr.magnitude

    if pr > 1:
        x0 = [0.05]

        vr = root(fun=_fun,
                  args=(Tr, pr),
                  x0=x0,
                  method='df-sane',
                  options={
                      'xatol': 1E-08,
                      'maxiter': 1500,
                      'disp': True,
                  })['x'][0]
    else:
        loss = lambda vr, Tr: (((pr * vr) / Tr) - leekesler(Tr=Tr, vr=vr)) ** 2
        vr = minimize_scalar(fun=loss,
                             args=(Tr),
                             method='bounded',  # bounded Brent optimizer
                             bounds=[0.2901, 1E+08],
                             options={'xatol': 1E-08,
                                      'maxiter': 1500,
                                      'disp': True, }
                             )['x']

    Z = leekesler(Tr=Tr, vr=vr)
    return Z


def hdep(pr: float, Tr: float, omega: float = 0, **coeffs) -> float:
    """
    Coefficients according to
    A Generalized Thermodynamic Correlation Based on Three-Parameter Corresponding States
    Lee & Kesler 1975
    p. 511
    Table I
    """
    # simple fluid
    b1 = 0.1181193
    b2 = 0.265728
    b3 = 0.154790
    b4 = 0.030323
    c1 = 0.0236744
    c2 = 0.0186984
    c3 = 0.0
    c4 = 0.042724
    d1 = 0.155488E-04  # 'd1': 0.155428E-04 (1978) OR 0.155488E-04 (VanWylen 8ed / Lee-Kesler 1975)
    d2 = 0.623689E-04
    beta = 0.65392
    gamma = 0.060167

    Z = get_Z(pr, Tr)
    vr = Z * Tr / pr

    term1 = (b2 + 2 * b3 / Tr + 3 * b4 / Tr **2) / (Tr * vr)
    term2 = (c2 - 3 * c3 / Tr ** 2)
    term3 = d2 / (5 * Tr * vr ** 5)
    term4 = (3 * c4 / (2 * Tr ** 3 * gamma)) * (beta + 1 - (beta + 1 + gamma / vr ** 2) * np.exp(-gamma / vr ** 2))
    hdep = -Tr * (Z - 1 - term1 - term2 + term3 + term4)
    return hdep


def sdep(pr: float, Tr: float, pc_atm: float, omega: float = 0, **coeffs) -> float:
    """
    Coefficients according to
    A Generalized Thermodynamic Correlation Based on Three-Parameter Corresponding States
    Lee & Kesler 1975
    p. 511
    Table I
    """
    # simple fluid
    b1 = 0.1181193
    b2 = 0.265728
    b3 = 0.154790
    b4 = 0.030323
    c1 = 0.0236744
    c2 = 0.0186984
    c3 = 0.0
    c4 = 0.042724
    d1 = 0.155428E-04  # 'd1': 0.155428E-04 (1978) OR 0.155488E-04 (VanWylen 8ed / Lee-Kesler 1975)
    d2 = 0.623689E-04
    beta = 0.65392
    gamma = 0.060167

    p = pr * pc_atm

    Z = get_Z(pr, Tr)
    vr = Z * Tr / pr

    term1 = (b1 + b3/Tr**2 + 2*b4/Tr**3) / vr
    term2 = (c1 - 2*c3/Tr**3)/(2*vr**2)
    term3 = d1 / (5*vr**5)
    term4 = (2 * c4 / (2 * Tr ** 3 * gamma)) * (beta + 1 - (beta + 1 + gamma / vr ** 2) * np.exp(-gamma / vr ** 2))

    sdep = -np.log(p / 1) + np.log(Z) - term1 - term2 - term3 - term4
    return sdep