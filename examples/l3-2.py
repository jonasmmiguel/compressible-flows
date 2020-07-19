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
    f = 0.016 * unit('')

    ARn = 2.5 * unit('')

    # ==========================================
    # given: operation parameters
    # ==========================================
    pb = 45 * unit('kPa')

    # ==========================================
    # analysis
    # ==========================================
    fld = f * L / D

    # state: i
    r_Ai_Aistar = ARn
    Mi = isen('M', A=r_Ai_Aistar, regime='supersonic')
    r_pc_pi = 1 / isen('p', M=Mi)
    r_pi_pstarf = fanno('p', M=Mi)
    fldi = fanno('fld', M=Mi)

    # state: s
    a = fldi - fld
    fld_s_min = max(0,  fldi - fld)     # TODO: problem is probably here
    fld_s_max = fldi
    Ms_options = [fanno('M', fld=fld_s_min, regime='supersonic'),
                  fanno('M', fld=fld_s_max, regime='supersonic')]

    configs = {'pc': [],
               'Me': [],
               'Ms': [],
               'deltaxsup/L': [],
               }
    for Ms in Ms_options:
        flds = fanno('fld', M=Ms)
        r_pstarf_ps = 1 / fanno('p', M=Ms)

        # state: s'
        Msl = nshock('Msl', Ms=Ms)
        r_ps_psl = 1 / nshock('p', Ms=Ms)
        r_psl_pstar = fanno('p', M=Msl)
        fldsl = fanno('fld', M=Msl)

        # state: e
        deltaxsup = (fldi - flds) * D / f
        deltaxsub = L - deltaxsup
        fld_deltaxsub = (f/D) * deltaxsub
        flde = fldsl - fld_deltaxsub
        Me = fanno('M', fld=flde, regime='subsonic')  # TODO: it was calculating sth even without specifying regime!!!
        r_pstar_pe = 1 / fanno('p', M=Me)
        r_pstar_pb = r_pstar_pe

        pc = r_pc_pi * r_pi_pstarf * r_pstarf_ps * r_ps_psl * r_psl_pstar * r_pstar_pb * pb

        # ==========================================
        # log results
        # ==========================================
        configs['pc'].append(pc)
        configs['Me'].append(Me)
        configs['Ms'].append(Ms)
        configs['deltaxsup/L'].append(deltaxsup/L)

    print('done')