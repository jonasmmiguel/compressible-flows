# Author: Jonas M. Miguel (jonasmmiguel@gmail.com)

import numpy as np
from scipy.optimize import minimize_scalar, minimize, fsolve, brentq

Runiv = 8.3145              # [J/mol.K]

m_mol_air = 28.9645E-03     # [kg/mol]
Rair = Runiv / m_mol_air    # [J/kg.K] gas constant for atmospheric air

m_mol_n2 = 28E-03           # [kg/mol]
Rn2 = Runiv / m_mol_n2      # [J/kg.K]

m_mol_co2 = 48E-03          # [kg/mol]
Rco2 = Runiv / m_mol_co2    # [J/kg.K]

m_mol_co = 32E-03           # [kg/mol]
Rco = Runiv / m_mol_co      # [J/kg.K]


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
            return isentropic('T', M=M, k=k) ** ( k / (k-1))

        elif output_type == 'A':
            return (1 / M) * ((1 + (0.5 * (k - 1)) * M ** 2) / (0.5 * (k + 1))) ** (
                    0.5 * (k + 1) / (k - 1))

    except KeyError:
        if input['A_ratio'] and output_type == 'M':
            loss = lambda M, k: (isentropic('A', M=M, k=k) - input['A_ratio'])**2       # squared error

            if input['regime'] == 'subsonic':
                M_range = [(0, 1)]
                M_initial_guess = 0.3
            elif input['regime'] == 'supersonic':
                M_range = [(1, 10)]
                M_initial_guess = 1.6

            M = minimize(loss, x0=np.array(M_initial_guess), bounds=M_range, method='TNC', args=k, options={'maxiter': 100, 'ftol': 1e-8})['x'][0]
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
            return (1 + k*Ms**2)/(1 + k*Msl**2)

    except KeyError:
        return NotImplementedError('Inverted relations for normal shock not yet implemented')
    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))


def fanno(output_type, **input):
    k = get_k(input)
    try:
        M = input['M']
        if output_type == 'T':
            return ((k+1)/2)*isentropic('T', M=M, k=k)

        elif output_type == 'p':
            T_ratio = fanno('T', M=M, k=k)
            return (1/M)*T_ratio**0.5
        elif output_type == 'pt':
            T_ratio = fanno('T', M=M, k=k)
            return (1/M)*T_ratio**( -(k+1)/(2*(k-1)))

        elif output_type == 'fld':
            T_ratio = fanno('T', M=M, k=k)
            term1 = ((k+1)/(2*k)) * np.log(T_ratio*M**2)
            term2 = (1 / k) * ((1 / M ** 2) - 1)
            return term1 + term2

    except KeyError:
        if input['fld'] and input['regime'] and output_type == 'M':
            loss = lambda M, k: (fanno('fld', M=M, k=k) - input['fld'])**2  # squared error

            if input['regime'] == 'subsonic':
                M_range = [(0, 1)]
                M_initial_guess = 0.3
            elif input['regime'] == 'supersonic':
                M_range = [(1, 10)]
                M_initial_guess = 1.6

            M = minimize(loss, x0=np.array(M_initial_guess), bounds=M_range, method='TNC', args=k, options={'maxiter': 100, 'ftol': 1e-8})['x'][0]
            return M

    else:
        return NotImplementedError('Unexpected input ({}), output ({}) configuration given.'.format(input, output_type))


def c(T, k=1.4, R=Rair):
    return (k*R*T)**(1/2)


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


if __name__ == '__main__':
    print('done')






