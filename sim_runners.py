"""This module contains functions to run the simulation under different sets of conditions to produce useful data."""


import math
from typing import List, Tuple

import numpy as np
import pandas as pd

import sim


def volume_sweep(events: int, 
                num_atoms: int, 
                energy: float, 
                mass: float, 
                radius: float, 
                data_points: int, 
                dpath: str = '..') -> None:
    """Call sim.run() across a range of volumes and output the resulting data as a csv."""

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
        point: Tuple[float, float, float] = sim.run(events, num_atoms, vol, energy, mass, radius)

        data.append(point)

    df: pd.DataFrame = pd.DataFrame(data=data,columns=["Pressure","Volume","Nkt"])
    df.to_csv(f"{dpath}/vsweep.csv")


def temp_sweep(events: int,
                num_atoms: int,
                volume: float,
                mass: float,
                radius: float,
                data_points: int,
                dpath: str = '..') -> None:

    """Call sim.run() across a range of temperatures and save the results as a CSV."""

    start_energy: float = num_atoms * mass
    end_energy = 1e6 * start_energy
    es: np.ndarray = np.linspace(start_energy, end_energy, num=data_points)

    data: List[Tuple[float, float, float]] = []

    for e in es:
        point: Tuple[float, float, float] = sim.run(events, num_atoms, volume, e, mass, radius)

        data.append(point)
    
    df = pd.DataFrame(data, columns=['Pressure', 'Volume', 'NkT'])
    df.to_csv(f"{dpath}/tsweep.csv")


