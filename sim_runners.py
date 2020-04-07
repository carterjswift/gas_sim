#this file contains methods to run the simulation for data gathering

import numpy as np
import pandas as pd
import time
import math
from typing import List, Tuple
from sim import run_sim

#sweeps across a range of volumes determined by the total particle volume
def volume_sweep(events: int, num_atoms: int, energy: float, mass: float, radius: float, data_points: int) -> None:
    part_vol: float = radius**(3) * np.pi * 4/3
    print(part_vol)
    start_vol: float = 4 * num_atoms * part_vol
    print(start_vol)
    end_vol: float = 10**10 * start_vol
    print(end_vol)

    vols: np.ndarray = np.logspace(math.log10(start_vol),math.log10(end_vol), data_points)
    print(vols)

    data: List[Tuple[float, float, float]] = []

    for vol in vols:
        point: Tuple[float, float, float] = run_sim(events, num_atoms, vol, energy, mass, radius)

        data.append(point)

    df: pd.DataFrame = pd.DataFrame(data=data,columns=["Pressure","Volume","Nkt"])
    df.to_csv('../ideal_gas.csv')

volume_sweep(10000,200,100,0.1,0.1,100)