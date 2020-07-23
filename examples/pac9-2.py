from src.workbench import isentropic as isen, nshock, fanno, rayleigh
from src.utils.materials import get_R_cp, unit
import numpy as np
import pandas as pd
import plotly.express as px

if __name__ == '__main__':
    # ==========================================
    # given: material model parameters
    # ==========================================
    k = 1.4
    [R, cp] = get_R_cp(material='AIR', k=k)

    # ==========================================
    # given: design parameters
    # ==========================================
    L = 6.1 * unit('m')
    D = 30.5 * unit('cm')
    f = 0.02 * unit('')

    ARn = 3.0 * unit('')

    # ==========================================
    # given: operation parameters
    # ==========================================
    pc = 690 * unit('kPa')  # chamber

    # ==========================================
    # analysis
    # ==========================================
    # state: i
    r_Ai_Aistar_isen = ARn
    Mi = isen('M', A=r_Ai_Aistar_isen, regime='supersonic')
    r_pi_pc = isen('p', M=Mi)
    r_pstarf_pi = 1 / fanno('p', M=Mi)
    fldi = fanno('fld', M=Mi)  # OK: fld2 = 0.42 < 1.07 = fld

    # state: s
    fld = f * L / D
    fld_s_min = fldi - fld
    fld_s_max = fldi
    Ms_options = [fanno('M', fld=fld_s_min, regime='supersonic'),
                  fanno('M', fld=fld_s_max, regime='supersonic')]

    configs = {'pe': [],
               'Me': [],
               'deltaxsup/L': [],
               }
    for Ms in Ms_options:
        # state: s
        r_ps_pstarf = fanno('p', M=Ms)
        flds = fanno('fld', M=Ms)

        # state: s'
        Msl = nshock('Msl', Ms=Ms)
        r_psl_ps = nshock('p', Ms=Ms)  # check: nshock 'p' ratio really implemented as psl/ps or ps/psl?
        r_pstarf_psl = 1 / fanno('p', M=Msl)
        fldsl = fanno('fld', M=Msl)

        # state: e
        deltaxsup = (fldi - flds) * D / f
        deltaxsub = L - deltaxsup
        fld_deltaxsub = (f/D) * deltaxsub
        flde = fldsl - fld_deltaxsub
        Me = fanno('M', fld=flde, regime='subsonic')
        r_pe_pstarf = fanno('p', M=Me)

        pe = r_pe_pstarf * r_pstarf_psl * r_psl_ps * r_ps_pstarf * r_pstarf_pi * r_pi_pc * pc

        # ==========================================
        # log results
        # ==========================================
        configs['pe'].append(pe)
        configs['Me'].append(Me)
        configs['deltaxsup/L'].append(deltaxsup/L)

    print(configs)
    print('done')