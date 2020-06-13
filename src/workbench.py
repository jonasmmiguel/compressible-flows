# Author: Jonas M. Miguel (jonasmmiguel@gmail.com)

import numpy as np
from scipy.optimize import minimize, fsolve

Runiv = 8.3145              # [J/mol.K]

m_mol_air = 28.9645E-03     # [kg/mol]
Rair = Runiv / m_mol_air    # [J/kg.K] gas constant for atmospheric air

m_mol_n2 = 28E-03           # [kg/mol]
Rn2 = Runiv / m_mol_n2      # [J/kg.K]

m_mol_co2 = 48E-03          # [kg/mol]
Rco2 = Runiv / m_mol_co2    # [J/kg.K]

m_mol_co = 32E-03           # [kg/mol]
Rco = Runiv / m_mol_co      # [J/kg.K]


def c(T, k=1.4, R=Rair):
    return (k*R*T)**(1/2)
# print( area_ratio(M=0.6476, k=1.4) )


def psi_T(M, k=1.4):  # T_ratio = T/Tt
    return 1/( 1 + 0.5*(k-1)*M**2 )


def psi_T_inv(T_ratio, k=1.4):
    """
    Determine the (unambiguous) solution for Mach Number, given T/Tt, gamma.

    Usage example: M_from_T_ratio(T_ratio=0.99356) -> 0.18

    :param k: gamma
    :param T_ratio: T/Tt
    :return:
    """
    return (( -1+1/T_ratio )*2/(k-1))**0.5


def psi_p(M, k=1.4):  # p_ratio = p/pt
    return psi_T(M, k) ** ((k-1) / k)


def psi_p_inv(p_ratio, k=1.4):
    """
    Determine the (unambiguous) solution for Mach Number, given p/pt, gamma.

    Usage example: # TODO

    :param k: gamma
    :param T_ratio: T/Tt
    :return:
    """
    T_ratio = p_ratio**( (k-1)/k )
    return psi_T_inv(T_ratio, k)


def psi_A(M, k=1.4):  # A/A*
    """
    Determine A/A* for a given M, gamma.

    :param M:
    :param k:
    :return:
    """
    return (1/M)*( (1 + (0.5*(k-1))*M**2) /( 0.5*(k+1) ))**( 0.5*(k+1)/(k-1) )   # Zucker p. 130, eq 5.37
# print( M_from_area_ratio(k=1.4, area_ratio=1.13790) )


def psi_A_inv(area_ratio, k, regime):
    """
    Determine sub- and supersonic solutions for Mach Number, given A/A*, gamma.

    Usage example: M_from_area_ratio(k=1.4, area_ratio=1.1379) -> [0.64755967 1.43999497]

    :param k: gamma
    :param area_ratio: A/A*
    :return:
    """
    f = lambda M, k: psi_A(M, k) - area_ratio
    [Msub, Msuper] = fsolve(f, x0=[0.1, 10], args=(k))
    if regime == 'subsonic':
        return Msub
    else:
        return Msuper


def psi_pA(M, k=1.4):  # pA/(pt A*)
    return psi_A(M, k)*psi_p(M, k)


def psi_pA_inv(PAR_given, k=1.4):   # determine the physically unambiguous M for a given pA/(pt A*)
    f = lambda M, k: psi_pA(M, k) - PAR_given
    M = fsolve(f, x0=[0.1, 10], args=(k))   # 2 solutions: 1 positive, 1 negative (physically unreasonable).

    # TODO: instead of fsolve, use minimize(bounds=...) to filter out negative values
    # TODO: verify if solver converges correctly with given initial guesses (problem detected on 6.4)
    # M = minimize(f, x0=[0.1, 10], args=(k)
    return M


def phi_pt(Ms, k):
    """
    Determine the total pressure ratio (pt_s'/pt_s) btw down- and upstream sections of a normal shock.

    :param Ms: upstream Mach Number (Ms)
    :param k: fluid heat capacity ratio
    :return:
    """
    term1a = ((k+1)/2)*Ms**2
    term1b = 1 + ((k-1)/2)*Ms**2
    term3 = (2*k/(k+1))*Ms**2 - ( (k-1)/(k+1) )
    return (term1a/term1b)**(k/(k-1)) * term3**(1/(1-k))


def phi_pt_inv(PR, k=1.4):
    f = lambda Ms, k: phi_pt(Ms, k) - PR
    # TODO: verify if solver converges correctly with given initial guesses (problem detected on 6.4)
    M = fsolve(f, x0=[1], args=(k))
    return M[0]


def phi_M(Ms, k):
    """
    Determine the Mach Number M_s' downstream of a normal shock.

    :param Ms: upstr, keam Mach Number (Ms)
    :param k: fluid heat capacity ratio
    :return: M_s'
    """
    term1 = Ms**2 + 2/(k-1)
    term2 = ( 2*k/(k-1) )*Ms**2 -1
    return (term1/term2)**(1/2)


def mdot_choking_per_A(pt, Tt, k=1.4, R=Rair):
    return pt*( (k/(R*Tt))*(2/(k+1))**((k+1)/(k-1)) )**0.5


def p1tilde(p1):
    ds = cp*np.log(T1/T0) - R*np.log(p1/p0)
    p1tilde = p0*(psi_p(M=0)/psi_p(M=M1))**(k/(k-1))*np.exp(ds/R)
    return p1tilde


if __name__=="__main__":
    # # Q2 P1
    # R = Rair
    # k = 1.4
    #
    # mdot = 13.5
    # T1 = 1033
    # u1 = 90
    # pamb = 1E+05
    # p2 = pamb
    # M2 = 1
    #
    # cp = (k/(k-1))*R
    # Tt = T1 + u1**2/(2*cp)
    #
    # T2 = Tt*psi_T(M2)
    #
    # c2 = (k*R*T2)**0.5
    # u2 = c2
    #
    # A2 = mdot/((p2/(R*T2))*u2)
    # F = p2*A2
    # p2tilde = 8266/A2
    #
    # F = (p2tilde+0.5*(p2tilde/(R*T2))*u2**2)*A2

    # A3Q1
    L1_to_L1max = 8.48341 * 0.0127 / (0.024*0.6096)
    print('done')






