from src.workbench import isentropic as isen
from src.workbench import nshock as ns
from src.workbench import fanno as f
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

    
    print('done')