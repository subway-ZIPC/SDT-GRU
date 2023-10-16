import torch

from trainer.train import train_model
from utils import get_logger

import yaml
import pandas as pd
import os


def run_model(times):
    log_dir = 'log/EX/HZ_single_TriMUL_{}'.format(times)
    logger = get_logger(log_dir)

    yaml_file = 'config/hz.yaml'
    with open(yaml_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg['device'] = 'cuda:0'
    torch.backends.cudnn.enabled = False
    # torch.cuda.set_device(0)
    # cfg['device'] = 'cpu'
    if cfg['train']['load_param'] == 'None':
        cfg['train']['load_param'] = None
    logger.info(cfg)

    test_result = train_model(cfg, logger, log_dir, seed=None)

    test_data = test_result[-1]
    test_data = test_data['MAE'] + test_data['RMSE'] + test_data['MAPE']

    for h in logger.handlers:
        logger.removeHandler(h)
    return test_data


if __name__ == '__main__':
    t = 5
    model_name = 'HZ_single_TriMUL'

    csv_name = os.path.join('./csv/', model_name + '_repeat_' + str(t) + '.csv')

    df_dict = {'NAME': [],
               'MAE-1': [], 'MAE-2': [], 'MAE-3': [], 'MAE-4': [],
               'RMSE-1': [], 'RMSE-2': [], 'RMSE-3': [], 'RMSE-4': [],
               'MAPE-1': [], 'MAPE-2': [], 'MAPE-3': [], 'MAPE-4': []}
    print("repeat " + str(t) + " times")
    for i in range(t):
        test_data = run_model(i)
        data = [model_name + '_' + str(i)] + test_data
        for idx, k in enumerate(df_dict.keys()):
            df_dict[k].append(data[idx])

    df_dict['NAME'].append(model_name + '_avg')
    for idx, k in enumerate(df_dict.keys()):
        if k != 'NAME':
            data = sum(df_dict[k]) / len(df_dict[k])
            df_dict[k].append(data)
    print(df_dict)
    df = pd.DataFrame(df_dict)
    df.to_csv(csv_name, index=False)
