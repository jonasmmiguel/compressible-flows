from src.workbench import isentropic as s
from src.workbench import nshock as ns
from src.workbench import fanno as f
import numpy as np

if __name__ == '__main__':
    pratio = isentropic('p', M=0.5)
    Tratio = isentropic('T', M=0.5)
    print('done')