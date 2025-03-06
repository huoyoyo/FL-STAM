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


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path, client_id):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path, client_id)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path, client_id)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path, client_id):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_' + str(client_id) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Client:
    DEFAULTS = {}

    def __init__(self, config, client_id, num_clients):
        # 初始化客户端参数
        self.__dict__.update(Client.DEFAULTS, **vars(config))  # 使用 vars() 转换为字典
        self.client_id = client_id
        self.num_clients = num_clients

        # 初始化模型
        self.build_model()

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

        self.device = torch.device(device)
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def get_model_parameters(self):
        noise_scale = 0.01  # 定义噪声的标准差
        model_params = {
            'AnomalyTransformer': self.model.state_dict(),  # 获取模型的状态字典
        }  # 为每个参数添加噪声
        for key in model_params.keys():
            for param_name in model_params[key]:
                noise = torch.randn_like(model_params[key][param_name]) * noise_scale  # 生成噪声
                model_params[key][param_name] += noise  # 将噪声加到模型参数上
        return model_params  # 返回带有噪声的模型参数

    def set_model_parameters(self, parameters):
        # 更新模型参数
        self.model.load_state_dict(parameters['AnomalyTransformer'])
        if self.client_id==0:
            logger.info('===============Saving checkpoints============')
            path = self.model_save_path
            if not os.path.exists(path):
                 os.makedirs(path)
            torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
         
        

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
        logger.info("======================TRAIN MODE======================")
        

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(tqdm(self.train_loader)):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(
                        my_wassdistance_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)).detach())) + torch.mean(
                        my_wassdistance_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach(),
                            series[u])))
                    prior_loss += (torch.mean(
                        my_wassdistance_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach())) + torch.mean(
                        my_wassdistance_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

                # # calculate Association discrepancy
                # series_loss = 0.0
                # prior_loss = 0.0
                # for u in range(len(prior)):
                #     series_loss += (torch.mean(my_kl_loss(series[u], (
                #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                    self.win_size)).detach())) + torch.mean(
                #         my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                            self.win_size)).detach(),
                #                    series[u])))
                #     prior_loss += (torch.mean(my_kl_loss(
                #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                 self.win_size)),
                #         series[u].detach())) + torch.mean(
                #         my_kl_loss(series[u].detach(), (
                #                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                #                                                                                        self.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path, client_id=self.client_id)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
