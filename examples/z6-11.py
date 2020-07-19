from src.workbench import isentropic as isen
from src.workbench import nshock
from src.workbench import fanno
import numpy as np
import pint;                                              # units
unit = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=True,
    system=None,
    auto_reduce_dimensions=True,)


Runiv = 8.3145 * unit('J / (mol * K)')

m_mol_air = 28.9645E-03 * unit('kg / mol')
Rair = ( Runiv / m_mol_air ) * unit('J / (kg * K)')    # [J/kg.K] gas constant for atmospheric air

m_mol_n2 = 28E-03 * unit('kg / mol')
Rn2 = ( Runiv / m_mol_n2 ) * unit('J / (kg * K)')

m_mol_co2 = 48E-03 * unit('kg / mol')
Rco2 = ( Runiv / m_mol_co2 ) * unit('J / (kg * K)')

m_mol_co = 32E-03 * unit('kg / mol')
Rco = ( Runiv / m_mol_co ) * unit('J / (kg * K)')


if __name__ == '__main__':
    # === L3Q2 ===
    # material model: N2
    k = 1.4
    R = Rn2

    # ==========================================
    # given: design parameters
    # ==========================================
    Me_dp = 2.6 * unit('')

    # ==========================================
    # given: operation parameters
    # ==========================================
    a = (3/4) * unit('')

    # ==========================================
    # analysis
    # ==========================================
    # geometric relations
    r_Ae_Aestardp = isen('A', M=Me_dp, regime='subsonic')
    r_Ae_At = r_Ae_Aestardp
    r_As_At = (1 + a * (r_Ae_At ** 0.5 - 1)) ** 2

    r_Ae_As = r_Ae_At * (1 / r_As_At)
    r_Ae_Asl = r_Ae_As

    # state: s
    r_As_Asstar = r_As_At
    Ms = isen('M', A=r_As_Asstar, regime='supersonic')

    r_ps_pst = isen('p', M=Ms)
    r_ps_pc = r_ps_pst

    # state: sl
    Msl = nshock('Msl', Ms=Ms)
    r_Asl_Aslstar = isen('A', M=Msl)

    r_ptsl_psl = 1 / isen('p', M=Msl)
    r_pte_psl = r_ptsl_psl

    r_psl_ps = nshock('p', Ms=Ms)

    # state: e
    r_Ae_Aestar = r_Ae_Asl * r_Asl_Aslstar
    Me = isen('M', A=r_Ae_Aestar, regime='subsonic')
    r_pe_pte = isen('p', M=Me)

    r_pe_pc = r_pe_pte * r_pte_psl * r_psl_ps * r_ps_pc

    # ==========================================
    # log results
    # ==========================================
    configs = {}
    configs['r_pe_pc'] = r_pe_pc
    configs['Ms'] = Ms
    configs['Me'] = Me

    print(configs)
    print('done')