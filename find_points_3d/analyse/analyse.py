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
from find_points.find_points import closest_points, farthest_points


def analyse(contours_dia_deformed, config):
    points_per_contour = config.find_mesh_points.points_per_contour
    results_dir = config.analyse_simulation.results_dir
    output_dir = config.analyse_simulation.output_dir
    results_dia_original = pd.read_csv(os.path.join(results_dir, 'results.csv'))
    results_sys = pd.read_csv(os.path.join(results_dir, 'results_sys.csv'))
    contours_dia_original, contours_sys, _ = read_points(config)
    combined = pd.DataFrame(
        columns=[
            'slice',
            'max_distance_dia_original',
            'min_distance_dia_original',
            'area_dia_original',
            'max_distance_dia_deformed',
            'min_distance_dia_deformed',
            'area_dia_deformed',
            'max_distance_sys',
            'min_distance_sys',
            'area_sys',
        ],
    )
    combined['slice'] = contours_dia_deformed.keys()

    for slice in contours_dia_deformed.keys():
        contours_dia_deformed_slice = contours_dia_deformed[slice].set_index('node')[['x', 'y', 'z']]
        combined = compute_distances(combined, results_sys, contours_dia_deformed_slice, results_dia_original, slice)
        contours_dia_deformed_slice = sort_points(
            points_per_contour, results_dia_original, contours_dia_deformed_slice, slice
        )
        combined.loc[combined['slice'] == slice, 'area_dia_deformed'], contours_dia_deformed_2d = compute_area(
            contours_dia_deformed_slice.values, plot=False
        )
        combined.loc[combined['slice'] == slice, 'area_dia_original'], contours_dia_original_2d = compute_area(
            contours_dia_original[slice], plot=False
        )
        combined.loc[combined['slice'] == slice, 'area_sys'], contours_sys_2d = compute_area(
            contours_sys[slice], plot=False
        )
        plot_results(contours_dia_deformed_2d, contours_dia_original_2d, contours_sys_2d, combined, slice, output_dir)

    combined.to_csv(os.path.join(output_dir, 'combined.csv'), index=False)

    area_difference = combined['area_dia_deformed'] - combined['area_sys']
    max_distance_difference = combined['max_distance_dia_deformed'] - combined['max_distance_sys']
    min_distance_difference = combined['min_distance_dia_deformed'] - combined['min_distance_sys']
    print('Area difference: ', area_difference.mean())
    print('Max distance difference: ', max_distance_difference.mean())
    print('Min distance difference: ', min_distance_difference.mean())

    return area_difference, max_distance_difference, min_distance_difference


def compute_distances(combined, results_sys, results_dia_deformed, results_dia_original, slice):
    farthest_point_1 = results_dia_original.loc[results_dia_original['slice'] == slice, 'farthest_point_1']
    farthest_point_2 = results_dia_original.loc[results_dia_original['slice'] == slice, 'farthest_point_2']
    closest_point_1 = results_dia_original.loc[results_dia_original['slice'] == slice, 'closest_point_1']
    closest_point_2 = results_dia_original.loc[results_dia_original['slice'] == slice, 'closest_point_2']
    combined.loc[combined['slice'] == slice, 'max_distance_dia_deformed'] = math.dist(
        results_dia_deformed.loc[farthest_point_1].values[0], results_dia_deformed.loc[farthest_point_2].values[0]
    )
    combined.loc[combined['slice'] == slice, 'min_distance_dia_deformed'] = math.dist(
        results_dia_deformed.loc[closest_point_1].values[0], results_dia_deformed.loc[closest_point_2].values[0]
    )
    combined.loc[combined['slice'] == slice, 'max_distance_dia_original'] = results_dia_original.loc[
        results_dia_original['slice'] == slice, 'max_distance'
    ]
    combined.loc[combined['slice'] == slice, 'min_distance_dia_original'] = results_dia_original.loc[
        results_dia_original['slice'] == slice, 'min_distance'
    ]
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


def plot_results(contours_dia_deformed, contours_dia_original, contours_sys, combined, slice, results_dir):
    # plot contours dia and contours_sys next to each other
    PADDING = 0.015
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    dia_og_x, dia_og_y = contours_dia_original.exterior.xy
    ax[0].plot(np.abs(dia_og_x), np.abs(dia_og_y))
    _, farthest_1, farthest_2 = farthest_points(contours_dia_original.exterior.coords)
    ax[0].plot(
        [np.abs(farthest_1[0]), np.abs(farthest_2[0])], [farthest_1[1], farthest_2[1]], linestyle='-', color='red'
    )
    _, closest_1, closest_2 = closest_points(contours_dia_original.exterior.coords)
    ax[0].plot(
        [np.abs(closest_1[0]), np.abs(closest_2[0])], [closest_1[1], closest_2[1]], linestyle='-', color='green'
    )
    ax[0].text(
        1 - PADDING,
        1 - PADDING,
        f"Area {combined.loc[combined['slice'] == slice, 'area_dia_original'].values[0]:.2f} mm²",
        transform=ax[0].transAxes,
        ha='right',
        va='top',
    )
    ax[0].set_title('Contours dia original')
    dia_x, dia_y = contours_dia_deformed.exterior.xy
    ax[1].plot(np.abs(dia_x), np.abs(dia_y))
    _, farthest_1, farthest_2 = farthest_points(contours_dia_deformed.exterior.coords)
    ax[1].plot(
        [np.abs(farthest_1[0]), np.abs(farthest_2[0])], [farthest_1[1], farthest_2[1]], linestyle='-', color='red'
    )
    _, closest_1, closest_2 = closest_points(contours_dia_deformed.exterior.coords)
    ax[1].plot(
        [np.abs(closest_1[0]), np.abs(closest_2[0])], [closest_1[1], closest_2[1]], linestyle='-', color='green'
    )
    ax[1].text(
        1 - PADDING,
        1 - PADDING,
        f"Area {combined.loc[combined['slice'] == slice, 'area_dia_deformed'].values[0]:.2f} mm²",
        transform=ax[1].transAxes,
        ha='right',
        va='top',
    )
    ax[1].set_title('Contours dia deformed')
    sys_x, sys_y = contours_sys.exterior.xy
    ax[2].plot(np.abs(sys_x), np.abs(sys_y))
    _, farthest_1, farthest_2 = farthest_points(contours_sys.exterior.coords)
    ax[2].plot(
        [np.abs(farthest_1[0]), np.abs(farthest_2[0])], [farthest_1[1], farthest_2[1]], linestyle='-', color='red'
    )
    _, closest_1, closest_2 = closest_points(contours_sys.exterior.coords)
    ax[2].plot(
        [np.abs(closest_1[0]), np.abs(closest_2[0])], [closest_1[1], closest_2[1]], linestyle='-', color='green'
    )
    ax[2].text(
        1 - PADDING,
        1 - PADDING,
        f"Area {combined.loc[combined['slice'] == slice, 'area_sys'].values[0]:.2f} mm²",
        transform=ax[2].transAxes,
        ha='right',
        va='top',
    )
    ax[2].set_title('Contours sys')
    x_lim = ax[0].get_xlim()
    y_lim = ax[2].get_ylim()
    y_range = y_lim[1] - y_lim[0]
    ax[0].set_ylim(y_lim)
    ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[0] + y_range)
    ax[2].set_xlim(x_lim)

    fig.savefig(os.path.join(results_dir, f'contours_slice_{slice}.png'))
