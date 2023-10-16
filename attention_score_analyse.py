import os

from model.DT_GRU import DT_GRU
from trainer.train import gen_data, move2device
from utils import StandardScaler, get_logger
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model, data_loader, scaler, device):
    model.eval()
    attn_list = []
    for idx, batch in enumerate(data_loader):
        x, y, *extras = batch
        x = scaler.transform(x)
        y = scaler.transform(y)
        x, y, *extras = move2device([x, y] + extras, device)

        with torch.no_grad():
            y_pred, attn = model(x, y, extras)
            attn_list.append(attn)

    attn_tensor = torch.concatenate(attn_list)
    return attn_tensor


def add_image(writer, image, text, step):
    if len(image.shape) == 2:
        fig, ax = plt.subplots()
        ax.matshow(image)
        writer.add_figure(text, fig, global_step=step)
    else:
        for i in range(image.shape[0]):
            add_image(writer, image[i], text + '|' + str(i), step)


def get_attention_score(model_yaml_file, dataset_yaml_file, weight_path, tensor_path, img_path, line):
    if not os.path.exists(tensor_path):
        with open(model_yaml_file, "r") as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(dataset_yaml_file, "r") as f:
            ds_cfg = yaml.load(f, Loader=yaml.FullLoader)

        model_cfg['device'] = 'cuda'
        # model_cfg['device'] = 'cpu'

        model = DT_GRU(model_cfg['model'])
        model.to(model_cfg['device'])
        model.load_state_dict(torch.load(weight_path))

        train_set, train_loader = gen_data(ds_cfg, 'train')
        test_set, test_loader = gen_data(ds_cfg, 'test')
        scaler = StandardScaler(mean=train_set.mean, std=train_set.std)

        attn_tensor = evaluate_model(model=model,
                                     data_loader=test_loader,
                                     scaler=scaler,
                                     device=model_cfg['device'])
        torch.save(attn_tensor, tensor_path)
    else:
        attn_tensor = torch.load(tensor_path)

    writer = SummaryWriter()

    for i in range(20):
        add_image(writer, attn_tensor[i], '', i)

    # total
    # print(*attn_tensor.shape[:-2])
    # attn_tensor_temp = attn_tensor.reshape(*attn_tensor.shape[:-2], -1)
    # attn_tensor_temp = attn_tensor_temp.reshape(-1, attn_tensor_temp.shape[-1])
    #
    # percent_list = []
    # for attn_i in attn_tensor_temp:
    #     count_total = [attn_i[torch.logical_and(line[i] < attn_i, attn_i < line[i + 1])].shape[0]
    #                    for i in range(len(line) - 1)]
    #     percent_total = [c / attn_i.shape[0] for c in count_total]
    #     percent_list.append(percent_total)
    # print(percent_list)
    # percent_list = np.array(percent_list)
    # print(percent_list.shape)
    # print(percent_list[:1000])
    # print(attn_tensor_temp.mean())
    #
    # plt.figure(figsize=(10, 6), dpi=1000)
    # plt.xlabel('range')
    # plt.ylabel('proportion')
    # plt.boxplot(percent_list)
    # plt.xticks(range(5), labels=[''] + ['[{},{}]'.format(line[i], line[i + 1]) for i in range(len(line) - 1)])
    # plt.savefig(img_path)
    # plt.show()


if __name__ == "__main__":
    model_yaml_file = './config/hz_attn_score.yaml'
    dataset_yaml_file = './config/hz_attn_score.yaml'
    weight_path = './log/HZ_DT_GRU/best.pt'
    tensor_path = './log/HZ_DT_GRU/test_attention_score.pth'
    img_path = './image/box.png'

    line = [0, 0.001, 0.01, 0.1, 1.0]
    print(line)
    get_attention_score(model_yaml_file, dataset_yaml_file, weight_path, tensor_path, img_path, line)
