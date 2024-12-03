import pickle
import os.path as osp
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from loader.utils import *


class BJMetro(Dataset):
    def __init__(self, cfgs, split):
        self.root = cfgs['root']
        self.num_nodes = 276
        self.num_features = 2
        self.in_len = 4
        self.out_len = 4
        self.num_intervals = 72
        self.start_time = '5:00'
        self.interval = 15
        self.eigenmaps_k = cfgs.get('eigenmaps_k', 8)
        self.similarity_delta = cfgs.get('similarity_delta', 0.1)
        self.split = split
        self.train_ratio = 0.72
        self.val_ratio = 0.08
        self.data = {}
        self.read_data_dyna()

        # train: 0.72, val: 0.08, test: 0.2
        self.restday_temp = pd.read_csv(osp.join(self.root, 'restday.csv'))
        self.restday = {}
        for i in range(self.restday_temp.shape[0]):
            time, rest = self.restday_temp.iloc[i]
            if time == '\t':
                break
            self.restday[time] = rest

        if split == 'train':
            complete_time_series = self.gen_complete_time_series()
            mean, std = self.compute_mean_std()
            origin_graph_conn = self.gen_graph_conn()  # provided by PVCGN
            graph_sml = self.gen_graph_sml(complete_time_series)
            # graph_sml_dtw = self.gen_graph_sml_dtw()  # provided by PVCGN
            # graph_cor = self.gen_graph_cor()  # provided by PVCGN

            graph_conn = get_k_hop_metrix(origin_graph_conn, 1e6)

            eigenmaps = self.gen_eigenmaps(graph_conn)
            # transition_matrices = self.gen_transition_matrices(graphs)
            scaled_laplacian = compute_scaled_laplacian(graph_conn)
            self.mean, self.std, self.complete_time_series, eigenmaps, scaled_laplacian = totensor(
                [mean, std, complete_time_series, eigenmaps, scaled_laplacian],
                dtype=torch.float32)
            # graphs = totensor({'graph_conn': graph_conn, 'graph_sml_dtw': graph_sml_dtw}, dtype=torch.float32)
            graphs = totensor({'graph_conn': graph_conn, 'graph_sml_dtw': graph_sml}, dtype=torch.float32)
            self.statics = {'eigenmaps': eigenmaps,
                            # 'transition_matrices': transition_matrices,
                            'graphs': graphs,
                            'scaled_laplacian': scaled_laplacian,
                            'adj_mx': origin_graph_conn}

    def read_data_dyna(self):
        with open(os.path.join(self.root, 'BEIJING_SUBWAY_15MIN.dyna'), 'r') as f:
            data = f.read().split('\n')

        tensor_dict = {}
        time_list = []
        for i, d in enumerate(data[1:]):
            row = d.split(',')
            if len(row) < 2:
                continue
            time, id, in_flow, out_flow = row[2:]
            id = int(id)
            in_flow = int(in_flow)
            out_flow = int(out_flow)
            if id == 0:
                time_list.append(time)
            if id in tensor_dict.keys():
                tensor_dict[id].append([in_flow, out_flow])
            else:
                tensor_dict[id] = [[in_flow, out_flow]]
        # print(len(time_list))
        # print(len(tensor_dict.keys()))
        tensor_np = []
        for key in tensor_dict:
            tensor_np.append(tensor_dict[key])
        tensor_np = np.array(tensor_np)
        tensor_np = tensor_np.transpose([1, 0, 2])
        time_np = np.array(time_list)

        tensor_np_roll = np.array([np.roll(tensor_np, -i, axis=0) for i in range(4)])
        tensor_np_roll = tensor_np_roll.transpose([1, 0, 2, 3])[:-3]
        time_np_roll = np.array([np.roll(time_np, -i, axis=0) for i in range(4)])
        time_np_roll = time_np_roll.transpose([1, 0])[:-3]

        tensor_np_roll_x = tensor_np_roll[:-4]
        tensor_np_roll_y = tensor_np_roll[4:]
        time_np_roll_x = time_np_roll[:-4]
        time_np_roll_y = time_np_roll[4:]
        # print(tensor_np_roll_x.shape)
        # print(tensor_np_roll_y.shape)
        # print(time_np_roll_x.shape)
        # print(time_np_roll_y.shape)

        data_num = tensor_np_roll_x.shape[0]
        train_num = int(data_num * self.train_ratio)
        val_num = int(data_num * self.val_ratio)
        test_num = data_num - train_num - val_num
        # print(data_num, train_num, val_num, test_num)
        line = {'train': [0, train_num], 'val': [train_num, train_num + val_num], 'test': [train_num + val_num, data_num]}
        line_split = line[self.split]

        self.data['x'] = tensor_np_roll_x[line_split[0]:line_split[1]]
        self.data['y'] = tensor_np_roll_y[line_split[0]:line_split[1]]
        self.data['xtime'] = time_np_roll_x[line_split[0]:line_split[1]]
        self.data['ytime'] = time_np_roll_y[line_split[0]:line_split[1]]

    def read_data_rel(self):
        with open(os.path.join(self.root, 'BEIJING_SUBWAY_15MIN.rel'), 'r') as f:
            rel_data = f.read().split('\n')

        graph_conn = np.zeros([self.num_nodes, self.num_nodes])
        for rel_row in rel_data[1:]:
            rel_row = rel_row.split(',')
            if len(rel_row) < 2:
                continue
            start_node, end_node, type = [int(i) for i in rel_row[2:5]]
            graph_conn[start_node, end_node] = type

        graph_conn = graph_conn + np.eye(self.num_nodes)
        graph_conn[graph_conn > 1] = 1.0
        return graph_conn

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, item):
        inputs = self.data['x'][item]
        targets = self.data['y'][item]

        inputs_time = self.time_transform(self.data['xtime'][item])
        targets_time = self.time_transform(self.data['ytime'][item])

        inputs_rest = self.rest_transform(self.data['xtime'][item])
        targets_rest = self.rest_transform(self.data['ytime'][item])

        inputs, targets = totensor([inputs, targets], dtype=torch.float32)
        inputs_time, targets_time, inputs_rest, targets_rest = totensor(
            [inputs_time, targets_time, inputs_rest, targets_rest], dtype=torch.int64)

        return inputs, targets, inputs_time, targets_time, inputs_rest, targets_rest

    def gen_complete_time_series(self):
        x, y = self.data['x'], self.data['y']
        num_samples = x.shape[0]  # number of samples
        m = self.num_intervals - self.in_len - self.out_len + 1  # number of samples in a day
        d = int(num_samples / m)  # number of days

        z = np.concatenate((x, y), axis=1)  # (num_samples, in_len + out_len, num_nodes, num_features)

        temp = [np.concatenate(
            (z[(u * m):((u + 1) * m):(self.in_len + self.out_len)].reshape(-1, self.num_nodes, self.num_features),
             z[((u + 1) * m - m % (self.in_len + self.out_len) + 1):((u + 1) * m), -1]), axis=0)
                for u in range(d)]
        complete_time_series = np.concatenate(temp, axis=0)  # (total_intervals, num_nodes, num_features)

        return complete_time_series

    def compute_mean_std(self):
        mean = self.data['x'].mean()
        std = self.data['x'].std()

        return mean, std

    def gen_graph_conn(self):
        # with open(osp.join(self.root, 'graph_hz_conn.pkl'), 'rb') as f:
        #     graph_conn = pickle.load(f).astype(np.float32)  # symmetric, with self-loops
        # return graph_conn
        return self.read_data_rel()

    def gen_graph_sml(self, complete_time_series):
        x = complete_time_series.transpose((1, 0, 2)).reshape(self.num_nodes, -1)
        graph_sml = compute_graph_sml(x, delta=self.similarity_delta)  # symmetric, with self-loops

        return graph_sml

    def gen_graph_sml_dtw(self):
        with open(osp.join(self.root, 'graph_bj_sml.pkl'), 'rb') as f:
            graph_sml_dtw = pickle.load(f).astype(np.float32)  # symmetric, with self-loops

        return graph_sml_dtw

    def gen_graph_cor(self):
        with open(osp.join(self.root, 'graph_bj_cor.pkl'), 'rb') as f:
            graph_cor = pickle.load(f).astype(np.float32)  # asymmetric, graph_cor[i, j] is the weight from j to i

        return graph_cor

    def gen_eigenmaps(self, graph_conn):
        eigenmaps = compute_eigenmaps(graph_conn, k=self.eigenmaps_k)

        return eigenmaps

    def gen_transition_matrices(self, graphs):
        # transform adjacency matrices (value span: 0.0~1.0, A(i, j) is the weight from j to i)
        # to transition matrices
        S_conn = row_normalize(add_self_loop(graphs['graph_conn']))
        S_sml = row_normalize(add_self_loop(graphs['graph_sml']))
        # S_cor = row_normalize(add_self_loop(graphs['graph_cor']))

        S = np.stack((S_conn, S_sml), axis=0)
        return S

    def time_transform(self, time):
        dt = [t.astype('datetime64[s]').astype(datetime) for t in time]
        hour, minute = [int(s) for s in self.start_time.split(':')]
        dt = [t.replace(hour=hour, minute=minute) for t in dt]
        dt = np.array([np.datetime64(t) for t in dt])
        time = np.array([np.datetime64(t) for t in time])
        time_ind = ((time - dt) / np.timedelta64(self.interval, 'm')).astype(np.int64)

        return time_ind

    def rest_transform(self, time):
        time = np.array([np.datetime64(t) for t in time])
        dt = [t.astype('datetime64[s]').astype(datetime) for t in time]
        dates = [t.strftime('%Y-%m-%d') for t in dt]
        rest_ind = np.array([self.restday[d] for d in dates]).flatten().astype(np.int64)  # 0: workday, 1: restday

        return rest_ind


if __name__ == '__main__':
    # HZ Train : 1188, Val : 132, Test: 330 - Total: 1650
    # train: 0.72, val: 0.08, test: 0.2

    # cfgs = yaml.safe_load(open('cfgs/HZMetro_MGT.yaml'))['dataset']
    # train_set = HZMetro(cfgs, split='train')
    # val_set = HZMetro(cfgs, split='val')
    # test_set = HZMetro(cfgs, split='test')
    # batch = train_set[0]

    import numpy as np
    import matplotlib.pyplot as plt

    path_15 = '../../data/BEIJING_SUBWAY/BEIJING_SUBWAY_15MIN/BEIJING_SUBWAY_15MIN.dyna'

    with open(path_15, 'r') as f:
        data = f.read().split('\n')
        # print(len(data))
        # print(type(data))

    tensor_dict = {}
    time_list = []
    for i, d in enumerate(data[1:]):
        # print(d)
        row = d.split(',')
        if len(row) < 2:
            continue
        time, id, in_flow, out_flow = row[2:]
        id = int(id)
        in_flow = int(in_flow)
        out_flow = int(out_flow)
        if id == 0:
            time_list.append(time)
        if id in tensor_dict.keys():
            tensor_dict[id].append([in_flow, out_flow])
        else:
            tensor_dict[id] = [[in_flow, out_flow]]
        # print(time, id, in_flow, out_flow)
    print(len(time_list))
    # print(tensor)
    print(len(tensor_dict.keys()))
    tensor_np = []
    for key in tensor_dict:
        tensor_np.append(tensor_dict[key])
    tensor_np = np.array(tensor_np)
    tensor_np = tensor_np.transpose([1, 0, 2])
    time_np = np.array(time_list)
    print(tensor_np.shape)
    print(time_np.shape)
    # print(tensor_np)
    print(time_np)

    # plt.figure()
    # plt.plot(tensor_np[:, 0, :])
    # plt.show()
