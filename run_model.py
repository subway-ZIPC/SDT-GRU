from trainer.train import train_model
from utils import get_logger

import yaml


if __name__ == '__main__':
    log_dir = 'log/HZ'
    logger = get_logger(log_dir)

    yaml_file = 'config/hz.yaml'
    with open(yaml_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['device'] = 'cuda'
    if cfg['train']['load_param'] == 'None':
        cfg['train']['load_param'] = None

    logger.info(cfg)

    train_model(cfg, logger, log_dir, seed=None)
