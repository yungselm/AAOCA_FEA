import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from itertools import combinations
from scipy.spatial import KDTree


def find_points(config, contours, mesh_coords, output_path, plot=False):
    
    if mesh_coords is not None:
        points_per_contour = config.find_mesh_points.points_per_contour
        points_cols = [f'point_{i}' for i in range(points_per_contour)]
        results = pd.DataFrame(
            columns=[
                'slice',
                'farthest_point_1',
                'farthest_point_2',
                'closest_point_1',
                'closest_point_2',
                'max_distance',
                'min_distance',
            ]
            + points_cols,
            index=contours.keys(),
        )
    else:
        results = pd.DataFrame(
            columns=[
                'slice',
                'max_distance',
                'min_distance',
            ],
            index=contours.keys(),
        )
    for slice, contour in contours.items():
        max_distance, farthest_point_1, farthest_point_2 = farthest_points(contour)
        min_distance, closest_point_1, closest_point_2 = closest_points(contour)

        if plot:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(projection='3d')
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                contour[:, 2],
                '-g',
                linewidth=2,
                label='Contour',
            )
            ax.plot(
                farthest_point_1[0],
                farthest_point_1[1],
                farthest_point_1[2],
                'bo',
                markersize=8,
                label='Farthest Point 1',
            )
            ax.plot(
                farthest_point_2[0],
                farthest_point_2[1],
                farthest_point_2[2],
                'ro',
                markersize=8,
                label='Farthest Point 2',
            )
            ax.plot(
                closest_point_1[0], closest_point_1[1], closest_point_1[2], 'go', markersize=8, label='Closest Point 1'
            )
            ax.plot(
                closest_point_2[0], closest_point_2[1], closest_point_2[2], 'ko', markersize=8, label='Closest Point 2'
            )
            plt.show()
            
        if mesh_coords is not None:
            downsampled_contour = contour[:: contour.shape[0] // points_per_contour + 1]
            downsampled_contour = np.apply_along_axis(
                find_point_in_mesh, axis=1, arr=downsampled_contour, mesh_coords=mesh_coords
            )

            results.loc[slice] = [
                slice,
                find_point_in_mesh(farthest_point_1, mesh_coords),
                find_point_in_mesh(farthest_point_2, mesh_coords),
                find_point_in_mesh(closest_point_1, mesh_coords),
                find_point_in_mesh(closest_point_2, mesh_coords),
                max_distance,
                min_distance,
                ] + downsampled_contour.tolist()
        else:
            results.loc[slice] = [
                slice,
                max_distance,
                min_distance,
            ]

    results = results.sort_values(by=['slice'])
    results.to_csv(output_path, index=False)


def farthest_points(contour):
    max_distance = 0
    farthest_points = None

    for point1, point2 in combinations(contour, 2):
        distance = math.dist(point1, point2)
        if distance > max_distance:
            max_distance = distance
            farthest_points = (point1, point2)

    return max_distance, farthest_points[0], farthest_points[1]


def closest_points(contour):
    num_points = len(contour)
    min_distance = math.inf
    closest_points = None

    index_1 = 0
    index_2 = num_points // 2

    while True:
        distance = math.dist(contour[index_1], contour[index_2])
        if distance < min_distance:
            min_distance = distance
            closest_points = (contour[index_1], contour[index_2])

        index_1 += 1
        index_2 += 1

        if index_1 >= num_points // 2:
            break

    return min_distance, closest_points[0], closest_points[1]


def find_point_in_mesh(point, mesh_coords):
    if not np.isnan(mesh_coords).any():
        _, index = KDTree(mesh_coords).query(point)
    else:
        logger.warning('Mesh coordinates contain NaN values.')
        index = np.nan
    return index + 1
