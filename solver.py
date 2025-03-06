import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from utils.SinkhornDistance import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from tqdm import tqdm
from loguru import logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    res = torch.mean(torch.sum(res, dim=-1), dim=1)
    return res


def my_wassdistance_loss(p, q):
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
    p = torch.transpose(p, 1, -2).to(device)
    q = torch.transpose(q, 1, -2).to(device)
    res, _, _ = sinkhorn(p, q)
    return res




class Solver(object):
    DEFAULTS = {}

    def __init__(self, config, clients, server):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)
        dataset = self.dataset
        log_path = os.path.join('log', dataset)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger.add(log_path)

        self.build_model()
        self.criterion = nn.MSELoss()

        self.clients = clients
        self.server = server

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(device)
            output, series, prior, _ = self.model(input)

            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(
                    my_wassdistance_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) +
                                torch.mean(
                                    my_wassdistance_loss(
                                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                                            1, 1, 1, self.win_size)).detach(), series[u])))

                prior_loss += (torch.mean(
                    my_wassdistance_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) +
                               torch.mean(
                                   my_wassdistance_loss(series[u].detach(), (
                                           prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1,
                                                                                                                  1,
                                                                                                                  1,
                                                                                                                  self.win_size)))))

            # # calculate Association discrepancy
            # series_loss = 0.0
            # prior_loss = 0.0
            # for u in range(len(prior)):
            #     series_loss += (torch.mean(my_kl_loss(series[u], (
            #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                    self.win_size)).detach())) + torch.mean(
            #         my_kl_loss(
            #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                     self.win_size)).detach(),
            #             series[u])))
            #     prior_loss += (torch.mean(
            #         my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                            self.win_size)),
            #                    series[u].detach())) + torch.mean(
            #         my_kl_loss(series[u].detach(),
            #                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
            #                                                                                            self.win_size)))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        self.federated_learning(num_rounds=2, local_epochs=2)

    def federated_learning(self, num_rounds, local_epochs):
        for round in range(num_rounds):

            print(f"=====================Round {round + 1}/{num_rounds}=========================")

            # 每个客户端进行局部训练
            for client in self.clients:
                client.train()  # 在这里调用客户端的训练方法

            # 聚合客户端模型参数
            print("=====================aggregated_params=========================")
            aggregated_params = self.server.aggregate()

            # 服务器将聚合后的模型参数广播到每个客户端
            print("=====================broadcast=========================")
            self.server.broadcast(aggregated_params)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))

        temperature = 50  # HY|?
        self.model.eval()
        with torch.no_grad():

            logger.info("======================TEST MODE======================")

            criterion = nn.MSELoss(reduce=False)

            # (1) stastic on the train set
            attens_energy = []
            for i, (input_data, labels) in enumerate(tqdm(self.train_loader)):
                input = input_data.float().to(device)
                output, series, prior, _ = self.model(input)
                loss = torch.mean(criterion(input, output), dim=-1)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_wassdistance_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_wassdistance_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_wassdistance_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_wassdistance_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature

                # # calculate Association discrepancy
                # series_loss = 0.0
                # prior_loss = 0.0
                # for u in range(len(prior)):
                #     if u == 0:
                #         series_loss = my_kl_loss(series[u], (
                #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                        self.win_size)).detach()) * temperature
                #         prior_loss = my_kl_loss(
                #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                     self.win_size)),
                #             series[u].detach()) * temperature
                #     else:
                #         series_loss += my_kl_loss(series[u], (
                #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                        self.win_size)).detach()) * temperature
                #         prior_loss += my_kl_loss(
                #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                     self.win_size)),
                #             series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)

            # (2) find the threshold
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_wassdistance_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_wassdistance_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_wassdistance_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_wassdistance_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature

                # # calculate Association discrepancy
                # series_loss = 0.0
                # prior_loss = 0.0
                # for u in range(len(prior)):
                #     if u == 0:
                #         series_loss = my_kl_loss(series[u], (
                #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                        self.win_size)).detach()) * temperature
                #         prior_loss = my_kl_loss(
                #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                     self.win_size)),
                #             series[u].detach()) * temperature
                #     else:
                #         series_loss += my_kl_loss(series[u], (
                #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                        self.win_size)).detach()) * temperature
                #         prior_loss += my_kl_loss(
                #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                     self.win_size)),
                #             series[u].detach()) * temperature

                # Metric
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
            print("Threshold :", thresh)
            logger.info("Threshold :", thresh)

            # (3) evaluation on the test set
            test_labels = []
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_wassdistance_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss = my_wassdistance_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_wassdistance_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach()) * temperature
                        prior_loss += my_wassdistance_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature

                # # calculate Association discrepancy
                # series_loss = 0.0
                # prior_loss = 0.0
                # for u in range(len(prior)):
                #     if u == 0:
                #         series_loss = my_kl_loss(series[u], (
                #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                        self.win_size)).detach()) * temperature
                #         prior_loss = my_kl_loss(
                #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                     self.win_size)),
                #             series[u].detach()) * temperature
                #     else:
                #         series_loss += my_kl_loss(series[u], (
                #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                        self.win_size)).detach()) * temperature
                #         prior_loss += my_kl_loss(
                #             (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                     self.win_size)),
                #             series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            test_labels = np.array(test_labels)

            scores = test_energy
            pred = (test_energy > thresh).astype(int)
            gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
