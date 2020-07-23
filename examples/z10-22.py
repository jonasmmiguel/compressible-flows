from src.workbench import isentropic as isen, nshock, fanno, rayleigh
from src.utils.materials import get_R_cp, unit
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize_scalar, basinhopping, brute


if __name__ == '__main__':
    # ==========================================
    # given: material model parameters
    # ==========================================
    k = 1.4
    [R, cp] = get_R_cp(material='AIR', k=k)

    # ==========================================
    # given: operation parameters
    # ==========================================
    pti = 172.4 * unit('kPa')
    Tti = 277.8 * unit('K')

    # # ==========================================
    # # CASE A
    # # ==========================================
    # # given
    # Mi = 0.3
    #
    # # analysis
    # r_pstarR_pi = 1 / rayleigh('p', M=Mi)
    # r_pi_pti = isen('p', M=Mi)
    # pb_max_chocking = pstarR = r_pstarR_pi * r_pi_pti * pti
    #
    # Ttstar = Tti / rayleigh('Tt', M=Mi)
    # Tte = Ttstar
    # qie = ( cp * (Tte - Tti) ).to('kJ/kg')

    # ==========================================
    # CASE B
    # ==========================================
    # given
    qie = 526 * unit('kJ/kg')
    pe = pb = 103.4 * unit('kPa')

    # analysis
    def determine_pe(Mi, pti, qie, cp):
        r_pstarR_pi = 1 / rayleigh('p', M=Mi)
        r_pi_pti = isen('p', M=Mi)

        Me = determine_Me(Mi, qie, cp)
        r_pe_pstarR = rayleigh('p', M=Me)
        pe_tilde = r_pe_pstarR * r_pstarR_pi * r_pi_pti * pti
        return pe_tilde

    def determine_Me(Mi, qie, cp):
        Tte = Tti + qie/cp
        r_Tti_TtstarR = rayleigh('Tt', M=Mi)
        r_Tte_TtstarR = (Tte / Tti) * r_Tti_TtstarR
        Me = rayleigh('M', regime='subsonic', Tt=r_Tte_TtstarR)
        return Me

    loss = lambda Mi, pti, qie, cp: ((pe - determine_pe(Mi=Mi, pti=pti, qie=qie, cp=cp)).magnitude) ** 2
    Mi = brute(func=loss,
              args=(pti, qie, cp),
              ranges=(slice(0.00, 1.00, 1E-03),),
              disp=True,
              finish=None,
              )

    Me = determine_Me(Mi, qie, cp)

    # display results
    configs = {'Mi': Mi,
               'Me': Me}
    print(configs)

    print('done')