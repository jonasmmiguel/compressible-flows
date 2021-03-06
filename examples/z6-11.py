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
    Me_dp = 2.6 * unit('')

    # ==========================================
    # given: operation parameters
    # ==========================================
    a = (3/4) * unit('')

    # ==========================================
    # analysis
    # ==========================================
    # geometric relations
    r_Ae_At = r_Ae_Aestardp = isen('A', M=Me_dp, regime='subsonic')
    r_As_At = (1 + a * (r_Ae_At ** 0.5 - 1)) ** 2

    r_Ae_Asl = r_Ae_As = r_Ae_At * (1 / r_As_At)

    # state: s
    r_As_Asstar = r_As_At
    Ms = isen('M', A=r_As_Asstar, regime='supersonic')

    r_ps_pc = r_ps_pst = isen('p', M=Ms)

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