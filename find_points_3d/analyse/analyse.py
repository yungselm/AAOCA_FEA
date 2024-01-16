import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
from shapely.geometry import Polygon

from find_points.read_points import read_points


def analyse(mesh_coords, config):
    points_per_contour = config.find_mesh_points.points_per_contour
    results_dir = config.analyse_simulation.results_dir
    output_dir = config.analyse_simulation.output_dir
    nodes_of_interest = pd.read_csv(os.path.join(results_dir, 'results.csv'))
    results_sys = pd.read_csv(os.path.join(results_dir, 'results_sys.csv'))
    _, contours_sys, _ = read_points(config)
    combined = pd.DataFrame(
        columns=[
            'slice',
            'max_distance_dia',
            'min_distance_dia',
            'area_dia',
            'max_distance_sys',
            'min_distance_sys',
            'area_sys',
        ],
    )
    combined['slice'] = mesh_coords.keys()

    for slice in mesh_coords.keys():
        new_coords_slice = mesh_coords[slice].set_index('node')[['x', 'y', 'z']]
        combined = compute_distances(combined, results_sys, new_coords_slice, nodes_of_interest, slice)
        new_coords_slice = sort_points(points_per_contour, nodes_of_interest, new_coords_slice, slice)
        combined.loc[combined['slice'] == slice, 'area_dia'], contours_dia_2d = compute_area(
            new_coords_slice.values, plot=True
        )
        combined.loc[combined['slice'] == slice, 'area_sys'], contours_sys_2d = compute_area(
            contours_sys[slice], plot=False
        )
        plot_results(contours_dia_2d, contours_sys_2d, slice, output_dir)

    combined.to_csv(os.path.join(output_dir, 'combined.csv'), index=False)

    area_difference = combined['area_dia'] - combined['area_sys']
    max_distance_difference = combined['max_distance_dia'] - combined['max_distance_sys']
    min_distance_difference = combined['min_distance_dia'] - combined['min_distance_sys']
    print('Area difference: ', area_difference.mean())
    print('Max distance difference: ', max_distance_difference.mean())
    print('Min distance difference: ', min_distance_difference.mean())

    return area_difference, max_distance_difference, min_distance_difference


def compute_distances(combined, results_sys, new_coords_slice, nodes_of_interest, slice):
    farthest_point_1 = nodes_of_interest.loc[nodes_of_interest['slice'] == slice, 'farthest_point_1']
    farthest_point_2 = nodes_of_interest.loc[nodes_of_interest['slice'] == slice, 'farthest_point_2']
    closest_point_1 = nodes_of_interest.loc[nodes_of_interest['slice'] == slice, 'closest_point_1']
    closest_point_2 = nodes_of_interest.loc[nodes_of_interest['slice'] == slice, 'closest_point_2']
    combined.loc[combined['slice'] == slice, 'max_distance_dia'] = math.dist(
        new_coords_slice.loc[farthest_point_1].values[0], new_coords_slice.loc[farthest_point_2].values[0]
    )
    combined.loc[combined['slice'] == slice, 'min_distance_dia'] = math.dist(
        new_coords_slice.loc[closest_point_1].values[0], new_coords_slice.loc[closest_point_2].values[0]
    )
    combined.loc[combined['slice'] == slice, 'max_distance_sys'] = results_sys.loc[
        results_sys['slice'] == slice, 'max_distance'
    ]
    combined.loc[combined['slice'] == slice, 'min_distance_sys'] = results_sys.loc[
        results_sys['slice'] == slice, 'min_distance'
    ]

    return combined


def sort_points(points_per_contour, nodes_of_interest, new_coords_slice, slice):
    """Sort points to generate correct polygons (else sorted w.r.t. node ID)"""
    contour_points = [f'point_{i}' for i in range(points_per_contour)]
    nodes = nodes_of_interest.loc[nodes_of_interest['slice'] == slice, contour_points].values[0]
    nodes = nodes[np.isin(nodes, new_coords_slice.index)]
    sorter = {node: i for i, node in enumerate(nodes)}
    new_coords_slice = new_coords_slice.sort_index(key=lambda x: x.map(sorter))
    new_coords_slice = new_coords_slice.loc[nodes]

    return new_coords_slice


def compute_area(coords, plot=False):
    points = Points(coords)
    plane = Plane.best_fit(points)
    if plot:
        plot_3d(
            points.plotter(c='k', s=50),
            plane.plotter(alpha=0.2),
        )
        plt.show()

    for i, point in enumerate(points):
        points[i] = plane.project_point(point)
    if plot:
        plot_3d(
            points.plotter(c='k', s=50),
            plane.plotter(alpha=0.2),
        )
        plt.show()

    norm_vector = plane.normal
    rotation_matrix = rotation_matrix_to_xy(norm_vector)

    for i, point in enumerate(points):
        point = np.dot(rotation_matrix, point.T).T
        points[i] = point - point[2] * np.array([0, 0, 1])

    if plot:
        plot_3d(
            points.plotter(c='k', s=50),
        )
        plt.show()

    polygon = Polygon(points[:, :2])

    return polygon.area, polygon


def rotation_matrix_to_xy(normal_vector):
    z_axis = normal_vector / np.linalg.norm(normal_vector)
    x_axis = np.cross([0, 0, 1], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.vstack((x_axis, y_axis, z_axis))
    return rotation_matrix


def plot_results(contours_dia, contours_sys, slice, results_dir):
    # plot contours dia and contours_sys next to each other
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    dia_x, dia_y = contours_dia.exterior.xy
    ax[0].plot(dia_x, dia_y)
    ax[0].set_title('Contours dia')
    sys_x, sys_y = contours_sys.exterior.xy
    ax[1].plot(sys_x, sys_y)
    ax[1].set_title('Contours sys')
    fig.savefig(os.path.join(results_dir, f'contours_slice_{slice}.png'))
