import glob

import numpy as np
import pandas as pd
from loguru import logger
from re import sub



def read_points(config):
    contours_dir = config.find_mesh_points.contours_dir
    contours_dir_sys = config.find_mesh_points.contours_dir_sys
    mesh_coords_dia = config.find_mesh_points.mesh_coords_file
    contour_files_dia = glob.glob(contours_dir + '/*.txt')
    contour_files_sys = glob.glob(contours_dir_sys + '/*.txt')

    contours_dia = read_contours(contour_files_dia)
    contours_sys = read_contours(contour_files_sys)    

    mesh_coords_dia = np.genfromtxt(mesh_coords_dia, delimiter='\t', usecols=(1, 2, 3), skip_header=1)

    return contours_dia, contours_sys, mesh_coords_dia

def read_contours(files):
    contours = {}
    for file in files:
        slice = int(sub('[^0-9]', '', file.split('/')[-1].split('.')[0]))
        contour = np.genfromtxt(file, delimiter=' ', comments='%')
        contours[slice] = contour
    
    return contours
