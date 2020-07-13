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
    # material model: Air
    k = 1.4
    R = Rair

    # ==========================================
    # given: design parameters
    # ==========================================
    L = 1 * unit('m')
    D = 1.5 * unit('cm')
    f = 1.6E-02 * unit('')

    ARn = 2.5 * unit('')

    # ==========================================
    # given: operation parameters
    # ==========================================
    pb = 45 * unit('kPa')

    # ==========================================
    # analysis
    # ==========================================
    # state: 2
    r_A2_Astar = ARn
    M2 = isen('M', A_ratio=r_A2_Astar, regime='supersonic')
    r_pt2_p2 = 1 / isen('p', M=M2)
    r_p2_pstar = fanno('p', M=M2)
    fld2 = fanno('fld', M=M2)

    # state: s
    Ms_options = [fanno('M', regime='supersonic', fld=(fanno('fld', M=M2) - (f * L / D))),
                  fanno('M', regime='supersonic', fld=(fanno('fld', M=M2)))]

    pt1_options = []
    for Ms in Ms_options:
        flds = fanno('fld', M=Ms)
        r_pstar_ps = 1 / fanno('p', M=Ms)

        # state: s'
        Msl = nshock('Msl', Ms=Ms)
        r_ps_psl = nshock('p', Ms=Ms)
        r_psl_pstar = fanno('p', M=Msl)
        fldsl = fanno('fld', M=Msl)

        # state: 3
        deltaxsup = (fld2 - flds) * D / f
        deltaxsub = L - deltaxsup
        fld_deltaxsub = (f/D) * deltaxsub
        fld3 = fldsl - fld_deltaxsub
        M3 = fanno('M', fld=fld3)
        r_pstar_p3 = 1 / fanno('p', M=M3)
        r_pstar_pb = r_pstar_p3

        pt2 = r_pt2_p2 * r_p2_pstar * r_pstar_ps * r_ps_psl * r_psl_pstar * r_pstar_pb * pb
        pt1 = pt2
        pt1_options.append(pt1)

    print('done')