# from src.workbench import isentropic as s
# from src.workbench import nshock as ns
# from src.workbench import fanno as f
import numpy as np

if __name__ == '__main__':
    # pratio = isentropic('p', M=0.5)
    # Tratio = isentropic('T', M=0.5)

    phi_R_Tt =       0.17355371
    Tt2_to_Tstar = (800.8/277.8) *  phi_R_Tt

    phi_s_p1 =       0.97249670
    phi_R_p2 =        1.98965246
    phi_R_p1 =         2.27272727
    pb = phi_s_p1 * phi_R_p1 * 172.4 / phi_R_p2

    # a = 277.8 + 523/1
    print('done')