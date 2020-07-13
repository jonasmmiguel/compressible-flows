from src.workbench import isentropic as isen
from src.workbench import nshock as ns
from src.workbench import fanno
from src.workbench import rayleigh as r
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
    #     pb = r('p', M=M2) * (1 / r('p', M=M1)) * isen('p', M=M1) * pt1
    #     eps = pb - pb_ref
    #     print('done')

    # ### 10.14
    # # Given
    # pt1 = 35E+05
    # Tt1 = 450
    # p1 = pt1
    # T1 = Tt1
    #
    # AR = 4.0
    # Tsl = 560
    #
    # # Calculations
    # M3 = isen('M', A_ratio=AR, regime='supersonic')
    # Tt3 = Tt1
    # T3 = Tt3 * isen('T', M=M3)
    # Tsl_to_Tstar = (Tsl/T3)*r('T', M=M3)
    # Msl = r('M', T=Tsl_to_Tstar, regime='subsonic')
    # Msl = 0.686
    # Ms = ns('Ms', Msl=Msl)
    #
    # Tt2 = Tsl/isen('T', M=Msl)
    # q = 1*(Tt2 - Tt1)

    ##### Q1_D12
    # Given
    k=1.33
    cp = 1.19
    R = cp*(k-1)/(k)

    pt1 = 3E+05
    Tt1 = 800 + 273.15
    pb = 1.013E+05

    D2 = 0.5
    D3 = 0.46
    L = 10.0

    # Calculations
    # p3 = isen('p', k=1.33, M=1) * pt1
    # is_blocked =  p3 > pb
    # M3 = 1
    #
    # T3 = Tt1 * isen('T', k=1.33, M=1)
    # mdot = M3 * ((k * R * T3) ** 0.5) * (p3 / (R * T3)) * (0.25 * np.pi * D3 ** 2)
    #
    # u3 = (k * R * T3) ** 0.5
    #
    # phi_A_M1 = isen('A', k=1.33, M=M3)
    # M2 = isen('M', k=1.33, regime='subsonic', A_ratio=((D2 / D3) ** 2))
    # M1 = M2
    # T1 = Tt1 * isen('T', k=1.33, M=M1)
    # u1 = (k * R * T1) ** 0.5
    # F = mdot * (u3 - u1)

    # Given
    L = 10
    f = 0.04
    D = D2

    # # Calculations
    M3 = 1
    T3 = Tt1 * isen('T', M=M3)
    M2 = isen('M', k=1.33, regime='subsonic', A_ratio=(isen('A', k=1.33, M=M3) * (D2 / D3) ** 2))
    fld2 = fanno('fld', M=M2)
    fld1 = f*L/D + fld2
    M1 = fanno('M', k=1.33, fld=fld1, regime='subsonic')
    p1 = pt1 * isen('p', k=1.33, M=M1)
    T1 = Tt1 * isen('T', k=1.33, M=M1)
    p3 = isen('p', k=1.33, M=M3) * (1 / isen('p', k=1.33, M=M2)) * fanno('p', k=1.33, M=M2) * (1/fanno('p', k=1.33, M=M1)) * p1
    is_blocked =  p3 > pb

    mdot = M3 * ((k * R * T3) ** 0.5) * (p3 / (R * T3)) * (0.25 * np.pi * D3 ** 2)
    u3 = (k * R * T3) ** 0.5
    u1 = (k * R * T1) ** 0.5
    F = mdot * (u3 - u1)

    print('done')