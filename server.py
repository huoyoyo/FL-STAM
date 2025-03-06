from utils.utils import *
from utils.SinkhornDistance import *
from model.AnomalyTransformer import AnomalyTransformer
from loguru import logger

class Server:
    def __init__(self, clients):
        self.clients = clients

    def aggregate(self):
        """聚合来自多个客户端的模型参数（使用简单的平均）"""
        num_clients = len(self.clients)
        aggregated_params = None

        # 假设所有客户端模型结构一致，获取第一个客户端的参数作为初始化
        for idx, client in enumerate(self.clients):
            client_params = client.get_model_parameters()
            if aggregated_params is None:
                aggregated_params = client_params
            else:
                for key in aggregated_params.keys():
                    aggregated_params[key] = {
                        param_name: aggregated_params[key][param_name] + client_params[key][param_name]
                        for param_name in aggregated_params[key]
                    }

        # 平均参数
        for key in aggregated_params.keys():
            aggregated_params[key] = {
                param_name: aggregated_params[key][param_name] / num_clients
                for param_name in aggregated_params[key]
            }

        return aggregated_params

    def broadcast(self, aggregated_params):
        """将聚合后的参数广播到每个客户端"""
        for client in self.clients:
            client.set_model_parameters(aggregated_params)
