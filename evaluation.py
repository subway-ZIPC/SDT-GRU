from model.SDT_GRUs import SDT_GRUs
from trainer.train import gen_data, evaluate_model
from utils import StandardScaler, get_logger

import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt


def get_pick_result(model_yaml_file, dataset_yaml_file, weight_path, log_dir):
    logger = get_logger(log_dir)
    logger.info("Model cfg : {} | Dataset cfg : {}".format(model_yaml_file, dataset_yaml_file))

    with open(model_yaml_file, "r") as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(dataset_yaml_file, "r") as f:
        ds_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # model_cfg['device'] = 'cuda'
    model_cfg['device'] = 'cpu'

    # for k, v in torch.load(weight_path, map_location=model_cfg['device']).items():
    #     print(k, v)

    model = SDT_GRUs(model_cfg['model'])
    model.to(model_cfg['device'])
    model.load_state_dict(torch.load(weight_path))

    train_set, train_loader = gen_data(ds_cfg, 'train')
    test_set, test_loader = gen_data(ds_cfg, 'test')
    scaler = StandardScaler(mean=train_set.mean, std=train_set.std)

    mae, mape, rmse, _ = evaluate_model(eval_type='train',
                                        model=model,
                                        data_loader=test_loader,
                                        scaler=scaler,
                                        device=model_cfg['device'],
                                        logger=logger)
    # print("AVG - MAE: {:.2f}, MAEP: {:.4f}, RMSE: {:.2f}".format(mae, mape, rmse))


if __name__ == "__main__":
    log_dir = './log/Temp'
    model_yaml_file = './config/hz.yaml'
    dataset_yaml_file = './config/hz.yaml'
    weight_path = './log/HZ_DT_GRU/best.pt'

    get_pick_result(model_yaml_file, dataset_yaml_file, weight_path, log_dir)
