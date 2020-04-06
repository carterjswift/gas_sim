#this file contains methods to run the simulation for data gathering

import numpy as np
import pandas as pd
import time
import math
from gas_sim import run_sim

#sweeps across a range of volumes determined by the total particle volume
def volume_sweep(events, num_atoms, energy, mass, radius, data_points):
    part_vol = radius**(3) * np.pi * 4/3
    print(part_vol)
    start_vol = 4 * num_atoms * part_vol
    print(start_vol)
    end_vol = 10**10 * start_vol
    print(end_vol)

    vols = np.logspace(math.log10(start_vol),math.log10(end_vol), data_points)
    print(vols)

    data = []

    for vol in vols:
        pressure, volume, NkT = run_sim(events, num_atoms, vol, energy, mass, radius)

        data.append([pressure, volume, NkT])

    df = pd.DataFrame(data=data,columns=["Pressure","Volume","Nkt"])
    df.to_csv('../ideal_gas.csv')

volume_sweep(10000,200,100,0.1,0.1,100)