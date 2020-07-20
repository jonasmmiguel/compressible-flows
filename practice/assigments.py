from src.workbench import isentropic as isen
from src.workbench import nshock as ns
from src.workbench import fanno as fn
import numpy as np

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
    # === A3Q1 ===
    # assumptions
    k = 1.4
    R = Rair

    # given
    M1 = 0.25
    pt1 = 1.38E+06
    Tt1 = 278
    D = 0.0508
    L = 0.6096
    f = 0.024

    # calculations
    A = np.pi*(D**2)/4

    T1 = isen('T', M=M1) * Tt1
    p1 = isen('p', M=M1) * pt1
    rho1 = p1/(R * T1)
    u1 = M1 * (k * R * T1)**0.5

    fld_2 = fn('fld', M=M1) - f*L/D
    M2 = fn('M', fld=fld_2, regime='subsonic')
    p2 = fn('p', M=M2) * (1 / fn('p', M=M1)) * (isen('p', M=M1)) * pt1
    Tt2 = Tt1
    T2 = Tt2 * isen('T', M=M2)
    rho2 = p2/(R * T2)
    u2 = M2 * (k * R * T2)**0.5

    Ff = A*( (p2 + rho2 * u2**2) - (p1 + rho1 * u1**2))

    # === A3Q2 ===
    # assumptions
    k = 1.4

    # given
    ARn = 2.5
    L = 13
    D = 1.5E-02
    f = 0.016
    pb = 45E+03

    # both cases
    Me = isen('M', A_ratio=ARn, regime='supersonic')

    # case p0max
    fld_e = fn('fld', M=Me)
    fld_s = fld_e - f*L/D
    Ms = fn('M', fld=fld_s, regime='supersonic')

    # case p0min
    Msl = ns('M', Ms=Me)
    fld_sl = fn('fld', M=Msl)
    fld_l = fld_sl - f*L/D
    Ml = fn('M', fld=fld_l, regime='subsonic')
    term1 = isen('p', M=Me) * ns('p', M=Me)
    p0min = isen('p', M=Me) * ns('p', M=Me) * fn('p', M=Msl) * fn('p', M=Ml)





    print('done')