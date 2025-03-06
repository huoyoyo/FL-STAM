import os
import argparse
import gc

from torch.backends import cudnn
from utils.utils import *

from solver import Solver
from loguru import logger

from client import Client
from server import Server

# import pdb


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)


    #     ===========Federated Learning=============
    # 初始化客户端
    num_clients = 6  # 例如，实例化6个客户端
    clients = []

    # 实例化客户端
    try:
        for client_id in range(num_clients):
            client = Client(config, client_id, num_clients)  # 传递客户端总数
            clients.append(client)  # 添加客户端到列表中
    except Exception as e:
        print(f"实例化客户端时出现错误: {e}")
    # 初始化服务器
    server = Server(clients)  # 直接将客户端列表传递给服务器

    # 初始化求解器
    solver = Solver(vars(config), clients, server)




    # solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=55)
    parser.add_argument('--output_c', type=int, default=55)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=20)
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/MSL')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=1)

    parser.add_argument("--noise_scale", type=float, default=0.05, help="噪声")

    config = parser.parse_args()

 
    log_path = os.path.join('log', config.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger.add(log_path)

    args = vars(config)
    logger.info('------------ Options -------------')
    for k, v in sorted(args.items()):
        logger.info('%s: %s' % (str(k), str(v)))
    logger.info('-------------- End ----------------')
    
    main(config)
