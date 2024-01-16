import os
import hydra

from omegaconf import DictConfig
from loguru import logger

from find_points.read_points import read_points
from find_points.find_points import find_points


@hydra.main(version_base=None, config_path='.', config_name='config')
def find_mesh_points(config: DictConfig) -> None:
    contours_dia, contours_sys, mesh_coords = read_points(config)
    output_path = os.path.join(config.analyse_simulation.results_dir, 'results.csv')
    output_sys = os.path.join(config.analyse_simulation.results_dir, 'results_sys.csv')
    find_points(config, contours_dia, mesh_coords, output_path, plot=False)
    find_points(config, contours_sys, None, output_sys, plot=False)


if __name__ == '__main__':
    find_mesh_points()
