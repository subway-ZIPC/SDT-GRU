from model.DT_GRU import DT_GRU
from trainer.train import gen_data, evaluate_model, move2device
from trainer import metrics
from utils import StandardScaler

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
import warnings
warnings.filterwarnings("ignore")

hztop20_index = np.array(
    [2,
     53,
     55,
     46,
     24,
     5,
     14,
     13,
     16,
     22,
     33,
     8,
     12,
     20,
     11,
     10,
     4,
     7,
     9,
     15])
shtop72_index = np.array([
    5,
    19,
    86,
    82,
    152,
    89,
    195,
    198,
    168,
    214,
    24,
    18,
    94,
    187,
    21,
    84,
    189,
    61,
    14,
    217,
    88,
    22,
    148,
    91,
    9,
    169,
    17,
    64,
    270,
    190,
    60,
    197,
    6,
    167,
    11,
    240,
    202,
    20,
    146,
    45,
    40,
    193,
    47,
    62,
    158,
    196,
    63,
    191,
    46,
    28,
    44,
    199,
    87,
    35,
    68,
    30,
    33,
    41,
    59,
    2,
    37,
    4,
    29,
    10,
    0,
    34,
    39,
    38,
    36,
    7,
    15,
    12
])

# 5:30 = 0
# time_pick : [7:30, 9:30], [17:30, 19:30]
# index     : [8   , 16  ], [48   , 56   ],
time_pick = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16,
                      48, 49, 50, 51, 52, 53, 54, 55, 56])


def evaluate_model(eval_type, model, data_loader, scaler, device, logger,
                   return_pred=False, model_name=False, statics=None, print_data=False,
                   pick_mode=0, pick_station=None):
    detail = True
    if eval_type == "val":
        detail = False

    model.eval()
    y_pred_list = []
    y_truth_list = []
    time_stamp_list = []
    for idx, batch in enumerate(data_loader):
        x, y, *extras = batch
        x = scaler.transform(x)
        y = scaler.transform(y)
        x, y, *extras = move2device([x, y] + extras, device)

        with torch.no_grad():
            y_pred, _ = model(x, y, extras)

            y = scaler.inverse_transform(y)
            y_pred = scaler.inverse_transform(y_pred)

            y_pred_list.append(y_pred.cpu().numpy())
            y_truth_list.append(y.cpu().numpy())
            time_stamp_list.append(extras[1].cpu().numpy())

    y_pred_np = np.concatenate(y_pred_list, axis=0)
    y_truth_np = np.concatenate(y_truth_list, axis=0)
    time_stamp = np.concatenate(time_stamp_list, axis=0)

    mae_list = []
    mape_list = []
    rmse_list = []
    mae_sum = 0
    mape_sum = 0
    rmse_sum = 0
    horizon = y_pred_np.shape[1]

    if print_data:
        print('-' * 50)
        print(model_name)
    for horizon_i in range(horizon):
        if pick_mode == 0:
            # pick time
            print('pick time')
            keep = np.in1d(time_stamp[:, horizon_i], time_pick)
            if not keep.any():
                continue
            y_truth_temp = y_truth_np[keep, horizon_i, :, :]
            y_pred_temp = y_pred_np[keep, horizon_i, :, :]
        elif pick_mode == 1:
            # pick station
            print('pick station')
            if pick_station is not None:
                y_truth_temp = y_truth_np[:, horizon_i, pick_station, :]
                y_pred_temp = y_pred_np[:, horizon_i, pick_station, :]
            else:
                raise 'Choose hztop20_index or shtop72_index for pick_station'
        elif pick_mode == 2:
            # pick time & station
            print('pick time & station')
            keep = np.in1d(time_stamp[:, horizon_i], time_pick)
            if not keep.any():
                continue
            if pick_station is not None:
                y_truth_temp = y_truth_np[:, horizon_i, pick_station, :]
                y_pred_temp = y_pred_np[:, horizon_i, pick_station, :]
            else:
                raise 'Choose hztop20_index or shtop72_index for pick_station'
            y_truth_temp = y_truth_temp[keep]
            y_pred_temp = y_pred_temp[keep]
        else:
            raise 'wrong pick mode, choose [0, 1, 2]'

        mae = metrics.masked_mae_np(y_pred_temp, y_truth_temp, null_val=0, mode='dcrnn')
        mape = metrics.masked_mape_np(y_pred_temp, y_truth_temp, null_val=0)
        rmse = metrics.masked_rmse_np(y_pred_temp, y_truth_temp, null_val=0)
        mae_sum += mae
        mape_sum += mape
        rmse_sum += rmse
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        msg = "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}"
        if print_data:
            print(msg.format(horizon_i + 1, mae, mape, rmse))

    if print_data:
        print('-' * 50)
    if detail:
        logger.info('Evaluation_{}_End:'.format(eval_type))

    return (mae_list, mae_sum), (mape_list, mape_sum), (rmse_list, rmse_sum)


def get_pick_result(model_info, pick_mode, pick_station=None):
    with open(model_info[0], "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg['device'] = 'cuda'

    model = DT_GRU(cfg['model'])
    model.to(cfg['device'])
    model.load_state_dict(torch.load(model_info[1]))

    train_set, train_loader = gen_data(cfg, 'train')
    test_set, test_loader = gen_data(cfg, 'test')
    scaler = StandardScaler(mean=train_set.mean, std=train_set.std)

    mae, mape, rmse = evaluate_model(eval_type='val',
                                     model=model,
                                     data_loader=test_loader,
                                     scaler=scaler,
                                     device=cfg['device'],
                                     logger=None,
                                     return_pred=True,
                                     model_name=model_info[-1],
                                     statics=None,
                                     print_data=True,
                                     pick_mode=pick_mode,
                                     pick_station=pick_station)


if __name__ == "__main__":
    yaml_file = './config/hz.yaml'
    weight_path = './log/HZ_origin/best.pt'
    pick_mode = 0
    pick_station = hztop20_index
    # pick_mode
    # 0 - pick time
    # 1 - pick station
    # 2 - pick time & station

    get_pick_result((yaml_file, weight_path, 'GCST'), pick_mode, pick_station)
