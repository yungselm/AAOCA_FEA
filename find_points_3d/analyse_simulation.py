import hydra

from omegaconf import DictConfig
from loguru import logger

from analyse.read_results import read_results
from analyse.analyse import analyse


@hydra.main(version_base=None, config_path='.', config_name='config')
def analyse_simulation(config: DictConfig) -> None:
    new_coords = read_results(config)
    area, max_d, min_d = analyse(new_coords, config)

if __name__ == '__main__':
    analyse_simulation()