from trainer.train import gen_data, move2device
from model.GCST import GCST
from utils import StandardScaler

import torch
import yaml


if __name__ == '__main__':
    yaml_file = './config/hz_test_graph_bias.yaml'
    weight_path = './log/HZ_test_graph_bias/best.pt'

    with open(yaml_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg['device'] = 'cuda'

    model = GCST(cfg['model'])
    model.to(cfg['device'])
    model.load_state_dict(torch.load(weight_path))

    train_set, train_loader = gen_data(cfg, 'train')
    test_set, test_loader = gen_data(cfg, 'test')
    scaler = StandardScaler(mean=train_set.mean, std=train_set.std)

    graph = [train_set.statics['graphs'][key] for key in train_set.statics['graphs']]
    graph = torch.stack(graph, dim=-1).to(cfg['device'])

    model.eval()
    for idx, batch in enumerate(test_loader):
        x, y, *extras = batch
        x = scaler.transform(x)
        y = scaler.transform(y)
        x, y, *extras = move2device([x, y] + extras, cfg['device'])

        with torch.no_grad():
            y_pred = model(x, y, graph, extras)

            y = scaler.inverse_transform(y)
            y_pred = scaler.inverse_transform(y_pred)
            break
