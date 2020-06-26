from src.workbench import isentropic as s
from src.workbench import nshock as ns
from src.workbench import fanno as f
from src.workbench import rayleigh as r
import numpy as np
from scipy.optimize import minimize_scalar, minimize, fsolve, brentq


def find_minimum(loss, input, k=1.4):
    if input['regime'] == 'subsonic':
        M_range = [(0, 1)]
        M_initial_guess = 0.98
    elif input['regime'] == 'supersonic':
        M_range = [(1, 10)]
        M_initial_guess = 1.02

    M1 = minimize(loss, x0=np.array(M_initial_guess), bounds=M_range, method='L-BFGS-B', args=k,
                  options={'maxiter': 10000, 'ftol': 1e-14})['x'][0]

    M2 = minimize(loss, x0=np.array(M_initial_guess), bounds=M_range, method='TNC', args=k,
                  options={'maxiter': 10000, 'ftol': 1e-14})['x'][0]
    return 0.5 * (M1 + M2)


if __name__ == '__main__':
    # ### Extra 1
    # Tt1 = 277.8
    # Tt2 = Tt1 + 523/1
    # pt1 = 172.4
    # pb_ref = 103.4
    #
    # for M1 in np.linspace(0.2, 0.3, 9):
    #     Tt2_to_Tstar = (Tt2 / Tt1) * r("Tt", M=M1)
    #     M2 = r('M', Tt=Tt2_to_Tstar, regime='subsonic')
    #     pb = r('p', M=M2) * (1 / r('p', M=M1)) * s('p', M=M1) * pt1
    #     eps = pb - pb_ref
    #     print('done')

    ### 10.14


    print('done')