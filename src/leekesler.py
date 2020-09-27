import numpy as np
from scipy.optimize import root, minimize_scalar, brute, fmin
from src.utils.webscrapping import get_crit_state
import pint;  # units
import pickle
from random import uniform

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

# reference fluid
b1_ref = 0.2026579
b2_ref = 0.331511
b3_ref = 0.027655
b4_ref = 0.203488
c1_ref = 0.0313385
c2_ref = 0.0503618
c3_ref = 0.016901
c4_ref = 0.041577
d1_ref = 0.48736E-04
d2_ref = 0.0740336E-04
beta_ref = 1.226
gamma_ref = 0.03754
omega_ref = 0.3978


# def init_unit_system():
unit = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=True,
    system=None,
    auto_reduce_dimensions=True, )
#
#     pickle.dump(unit, open('./utils/unit.pkl', 'wb'))
#
#
# init_unit_system()
# unit = pickle.load(open('./utils/unit.pkl', 'rb'))


def BWR(Tr, vr):
    """
    Calculate Z via the Benedict-Webb-Rubin Equation using simple fluid coefficients

    :param vr:
    :param Tr:
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


def BWR_ref(Tr, vr):
    """
    Calculate Z via the Benedict-Webb-Rubin Equation using reference fluid coefficients

    :param vr:
    :param Tr:
    :return: Z: compressibility factor
    """
    B = b1_ref - (b2_ref / Tr) - (b3_ref / Tr ** 2) - (b4_ref / Tr ** 3)
    C = c1_ref - (c2_ref / Tr) + (c3_ref / Tr ** 3)
    D = d1_ref + (d2_ref / Tr)

    if abs(vr) < 1e-08:
        vr = 1e-08

    tail_term = (c4_ref / (Tr ** 3 * vr ** 2)) * (beta_ref + gamma_ref / vr ** 2) * np.exp(-gamma_ref / vr ** 2)

    Z = 1 + (B / vr) + (C / vr ** 2) + (D / vr ** 5) + tail_term
    return Z


def acentric_factor(fluid):
    get_crit_state(fluid, unit)
    Tc, pc = fluid.critical()
    pref = fluid.ps( 0.7 * Tc )
    pref_r = pref / pc
    omega = - np.log(pref_r) / np.log(10) - 1
    return omega


def leekesler(Tr: float, vr: float, omega=0.0) -> float:
    Z0 = BWR(Tr, vr)
    Z = Z0
    if omega != 0:
        Zref = BWR_ref(Tr, vr)
        Z = Z0 + (omega / omega_ref) * (Zref - Z0)
    return Z


def _fun(vr, Tr, pr):
    return (pr * vr) / Tr - leekesler(Tr=Tr, vr=vr)


def _loss(vr, Tr, pr):
    return (((pr * vr) / Tr) - leekesler(Tr=Tr, vr=vr)) ** 2


def validate_Tr_pr(Tr, pr):
    if type(pr) not in [float, int]:
        pr = pr.magnitude

    if type(Tr) not in [float, int]:
        Tr = Tr.magnitude

    return Tr, pr


def find_vr_via_brute(pr: float, Tr: float, polishing_optimizer=fmin):
    vmin=1E-2
    vmax=10
    n_gridpoints=1E+5
    grid_resolution = (vmax - vmin) / n_gridpoints
    vr = brute(
        func=_loss,                         # Squared Error of vr_ist, vr_soll
        ranges=(slice(vmin, vmax, grid_resolution),),
        args=(Tr, pr,),
        disp=True,
        finish=polishing_optimizer,         # polishing function
        workers=4
    )
    return vr


def find_vr_via_root(pr: float, Tr: float, vr0: float):
    vr = root(fun=_fun,
              args=(Tr, pr),
              x0=[vr0],
              method='df-sane',
              options={
                  'xatol': 1E-08,
                  'maxiter': 1500,
                  'disp': True,
              })['x'][0]
    return vr


def get_Z(pr: float, Tr: float) -> float:
    Tr, pr = validate_Tr_pr(Tr, pr)

    is_phasechange_possible = (Tr < 1.2) and (pr < 1)
    is_danger_zone = (Tr < 1.2) or (pr < 1)
    vr0 = 0.05
    polishing_optimizer = fmin

    if not is_danger_zone:
        vr = find_vr_via_root(pr=pr, Tr=Tr, vr0=vr0)
        Z = pr * vr / Tr
    elif not is_phasechange_possible:
        vr = find_vr_via_brute(pr=pr, Tr=Tr, polishing_optimizer=polishing_optimizer)
        Z = pr * vr / Tr
    else:
        Z, is_saturation_implausible = get_ambiguous_Z(Tr, pr)
        if is_saturation_implausible:
            vr = find_vr_via_brute(pr=pr, Tr=Tr, polishing_optimizer=polishing_optimizer)
            Z = pr * vr / Tr
    return Z


def get_delta_e(e):
    delta_e = [(e[i + 1] - e[i]) / e[i] for i in range(len(e) - 1)]
    return delta_e


def count_crossings(delta_e):
    is_delta_pos = np.array(delta_e) > 1E-6
    n_crossings = sum([is_delta_pos[i] != is_delta_pos[i + 1] for i in range(len(delta_e) - 1)])
    return n_crossings


def get_ambiguous_Z(Tr, pr):
    vmin = 1E-2
    vmax = 10
    n_gridpoints = 1E+5
    n_crossings = 0
    while n_crossings < 5 and vmax < 1000:
        grid_resolution = (vmax - vmin) / n_gridpoints
        _, _, vr, e = brute(
            func=_loss,  # Squared Error of vr_ist, vr_soll
            ranges=(slice(vmin, vmax, grid_resolution),),
            args=(Tr, pr,),
            disp=True,
            finish=None,  # polishing function
            workers=4,
            full_output=True
        )
        delta_e = get_delta_e(e)

        vrf = next(vr_ for vr_, de_ in zip(vr, delta_e) if de_ > 1E-6)
        vrg = next(vr_ for vr_, de_ in zip(vr[::-1], delta_e[::-1]) if de_ < 1E-6)

        n_crossings = count_crossings(delta_e=get_delta_e(e))
        vmax = vmax * 10

    Z = np.array([vrf, vrg]) * pr / Tr
    is_saturation_implausible = vmax > 1000

    return Z, is_saturation_implausible


def hdep(pr: float, Tr: float, omega: float = 0, **coeffs) -> float:
    Z = get_Z(pr, Tr)
    vr = Z * Tr / pr

    term1 = (b2 + 2 * b3 / Tr + 3 * b4 / Tr **2) / (Tr * vr)
    term2 = (c2 - 3 * c3 / Tr ** 2)
    term3 = d2 / (5 * Tr * vr ** 5)
    term4 = (3 * c4 / (2 * Tr ** 3 * gamma)) * (beta + 1 - (beta + 1 + gamma / vr ** 2) * np.exp(-gamma / vr ** 2))
    hdep = -Tr * (Z - 1 - term1 - term2 + term3 + term4)
    return hdep


def sdep(pr: float, Tr: float, pc_atm: float, omega: float = 0, **coeffs) -> float:
    p = pr * pc_atm

    Z = get_Z(pr, Tr)
    vr = Z * Tr / pr

    term1 = (b1 + b3/Tr**2 + 2*b4/Tr**3) / vr
    term2 = (c1 - 2*c3/Tr**3)/(2*vr**2)
    term3 = d1 / (5*vr**5)
    term4 = (2 * c4 / (2 * Tr ** 3 * gamma)) * (beta + 1 - (beta + 1 + gamma / vr ** 2) * np.exp(-gamma / vr ** 2))

    sdep = -np.log(p / 1) + np.log(Z) - term1 - term2 - term3 - term4
    return sdep


if __name__=='main':
    get_Z(.25, .8)
