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
    r_A2_Astar = ARn  # TODO: ensure I didnt interpreted this star as star_fanno
    M2 = isen('M', A=r_A2_Astar, regime='supersonic')
    r_pt2_p2 = 1 / isen('p', M=M2)
    r_p2_pstar = fanno('p', M=M2)
    fld2 = fanno('fld', M=M2)  # OK: fld2 = 0.42 < 1.07 = fld

    # state: s
    fld = f * L / D
    fld_s_min = fld2 - fld
    fld_s_max = fld2
    Ms_options = [fanno('M', fld=fld_s_min, regime='supersonic'),
                  fanno('M', fld=fld_s_max, regime='supersonic')]

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
        M3 = fanno('M', fld=fld3, regime='subsonic')  # TODO: it was calculating sth even without specifying regime!!!
        r_pstar_p3 = 1 / fanno('p', M=M3)
        r_pstar_pb = r_pstar_p3

        pt2 = r_pt2_p2 * r_p2_pstar * r_pstar_ps * r_ps_psl * r_psl_pstar * r_pstar_pb * pb
        pt1 = pt2
        pt1_options.append(pt1)

    print('done')