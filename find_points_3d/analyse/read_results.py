import glob
import os

import numpy as np
import pandas as pd
from loguru import logger
from re import findall



def read_results(config):
    results_dir = config.analyse_simulation.results_dir
    results_files = glob.glob(f'{results_dir}/s*_*.txt')

    current_slice = 0
    data = None
    new_coords = {}
    for file in results_files:
        new_slice = int(findall(r'\d+', os.path.basename(file))[0])
        if current_slice < new_slice:
            current_slice = new_slice
        if 'x' in os.path.splitext(os.path.basename(file))[0]:
            data = pd.read_csv(file, delimiter='\t', skiprows=1, names=['node', 'x', 'y', 'z', 'deformation'])
            data['x'] += data['deformation']
        elif 'y' in os.path.basename(file):
            data_y = pd.read_csv(file, delimiter='\t', skiprows=1, names=['node', 'x', 'y', 'z', 'deformation'])
            data['y'] += data_y['deformation']
        elif 'z' in os.path.basename(file):
            data['z'] += pd.read_csv(file, delimiter='\t', skiprows=1, names=['node', 'x', 'y', 'z', 'deformation'])['deformation']
        new_coords[current_slice] = data

    return new_coords