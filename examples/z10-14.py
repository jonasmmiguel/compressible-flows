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
    ARn = 4.0 * unit('')

    # ==========================================
    # given: operation parameters
    # ==========================================
    pti = 35E+05 * unit('N/m²')
    Tti = 450 * unit('K')

    Tsl = 560 * unit('K')

    # ==========================================
    # analysis
    # ==========================================
    # state e
    Me = isen('M', A=ARn, regime='supersonic')
    Tte = Tti
    Te = Tte * isen('T', M=Me)
    r_TtstarR_Tte = 1 / rayleigh('Tt', M=Me)
    r_Te_TstarR = rayleigh('T', M=Me)

    # state s'
    r_Tsl_Te = Tsl / Te
    r_Tsl_TstarR = r_Tsl_Te * r_Te_TstarR
    Msl = rayleigh('M', T=r_Tsl_TstarR, regime='subsonic')

    # state s
    Ms = nshock('Ms', Msl=Msl)
    r_Tts_TtstarR = rayleigh('Tt', M=Ms)
    Tts = r_Tts_TtstarR * r_TtstarR_Tte * Tte

    # heat transfer
    qes = (cp * (Tts - Tte)).to('kJ/kg')

    # ==========================================
    # log results
    # ==========================================
    configs = {}
    configs['Me'] = Me

    print(configs)
    print('done')