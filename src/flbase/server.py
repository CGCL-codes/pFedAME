from ..utils import autoassign, save_to_pkl, access_last_added_element, calculate_model_size
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from collections import OrderedDict


class Server:
    def __init__(self, server_config, clients_dict, **kwargs):
        """
        """
        autoassign(locals())
        self.server_model_state_dict = None
        self.server_model_state_dict_best_so_far = None
        self.num_clients = len(self.clients_dict)
        self.strategy = None
        self.average_train_loss_dict = {}
        self.average_train_acc_dict = {}
        # global model performance
        self.gfl_test_loss_dict = {}
        self.gfl_test_acc_dict = {}
        # local model performance (averaged across all clients)
        self.average_pfl_test_loss_dict = {}
        self.average_pfl_test_acc_dict = {}
        self.active_clients_indicies = None
        self.rounds = 0
        # create a fake client on the server side; use for testing the performance of the global model
        # trainset is only used for creating the label distribution
        self.server_side_client = kwargs['client_cstr'](
            kwargs['server_side_criterion'],
            kwargs['global_trainset'],
            kwargs['global_testset'],
            kwargs['server_side_client_config'],
            -1,
            kwargs['server_side_client_device'],
            **kwargs)

    def select_clients(self, ratio):
        assert ratio > 0.0, "Invalid ratio. Possibly the server_config['participate_ratio'] is wrong."
        num_clients = int(ratio * self.num_clients)
        selected_indices = np.random.choice(range(self.num_clients), num_clients, replace=False)
        return selected_indices

    def testing(self, round, active_only, **kwargs):
        raise NotImplementedError

    def collect_stats(self, stage, round, active_only, **kwargs):
        raise NotImplementedError()

    # def collect_loss_and_acc(self, stage, round, active_only=True):
    #     """
    #         No actual training and testing is performed. Just collect stats.
    #         stage: str;
    #             {"train", "test"}
    #         active_only: bool;
    #             True: compute loss and acc on active clients only
    #             False: compute loss and acc on all clients
    #     """
    #     client_indices = self.clients_dict.keys()
    #     if active_only:
    #         client_indices = self.active_clients_indicies
    #     total_loss = 0.0
    #     total_acc = 0.0
    #     total_samples = 0
    #     if stage == 'train':
    #         for cid in client_indices:
    #             client = self.clients_dict[cid]
    #             loss, acc, num_samples = client.train_loss_dict[round][-1], client.train_acc_dict[round][-1], client.num_train_samples
    #             total_loss += loss * num_samples
    #             total_acc += acc * num_samples
    #             total_samples += num_samples
    #         average_loss, average_acc = total_loss / total_samples, total_acc / total_samples
    #         self.average_train_loss_dict[round] = average_loss
    #         self.average_train_acc_dict[round] = average_acc
    #     else:
    #         # test stage
    #         # get global model performance
    #         self.average_test_loss_dict[round] = self.server_side_client.test_loss_dict[round][-1]
    #         self.average_test_acc_dict[round] = self.server_side_client.test_acc_dict[round][-1]
    #         if self.mode == "PFL":
    #             weights = self.clients_dict[client_indices[0]].test_pfl_loss_dict[round].keys()
    #             self.average_pfl_test_loss_dict[round] = {}
    #             self.average_pfl_test_acc_dict[round] = {}
    #             for weight in weights:
    #                 total_loss = 0.0
    #                 total_acc = 0.0
    #                 total_samples = 0
    #                 for cid in client_indices:
    #                     client = self.clients_dict[cid]
    #                     loss, acc, num_samples = client.test_pfl_loss_dict[round][weight][-1], client.test_pfl_acc_dict[round][weight][-1], client.num_train_samples
    #                     total_loss += loss * num_samples
    #                     total_acc += acc # use different scaling here
    #                     total_samples += num_samples
    #                 average_loss, average_acc = total_loss / total_samples, total_acc / len(client_indices)
    #                 self.average_pfl_test_loss_dict[round][weight] = average_loss
    #                 self.average_pfl_test_acc_dict[round][weight] = average_acc

    def aggregate(self, client_uploads):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def save(self, filename, keep_clients_model=False):
        if not keep_clients_model:
            for client in self.clients_dict.values():
                client.model = None
                client.trainloader = None
                client.trainset = None
                client.new_state_dict = None
        self.server_side_client.trainloader = None
        self.server_side_client.trainset = None
        self.server_side_client.testloader = None
        self.server_side_client.testset = None
        save_to_pkl(self, filename, self.run_id)

    def summary_setup(self):
        info = "=" * 60 + "Run Summary" + "=" * 60
        info += "\nDataset:\n"
        info += f" dataset:{self.server_config['dataset']} | num_classes:{self.server_config['num_classes']}"
        partition = self.server_config['partition']
        info += f" | partition:{self.server_config['partition']}"
        if partition == 'iid-equal-size':
            info += "\n"
        elif partition in ['iid-diff-size', 'noniid-label-distribution']:
            info += f" | beta:{self.server_config['beta']}\n"
        elif partition == 'noniid-label-quantity':
            info += f" | num_classes_per_client:{self.server_config['num_classes_per_client']}\n "
        else:
            if 'shards' in partition.split('-'):
                pass
            else:
                raise ValueError(f" Invalid dataset partition strategy:{partition}!")
        info += "Server Info:\n"
        info += f" strategy:{self.server_config['strategy']} | num_clients:{self.server_config['num_clients']} | num_rounds: {self.server_config['num_rounds']}"
        info += f" | participate_ratio:{self.server_config['participate_ratio']} | drop_ratio:{self.server_config['drop_ratio']}\n"
        info += f"Clients Info:\n"
        client_config = self.clients_dict[0].client_config
        info += f" model:{client_config['model']} | num_epochs:{client_config['num_epochs']} | batch_size:{client_config['batch_size']}"
        info += f" | optimizer:{client_config['optimizer']} | inint lr:{client_config['learning_rate']} | lr scheduler:{client_config['lr_scheduler']}\n"
        print(info)
        mdict = self.server_side_client.get_params()
        try:
            print(f" {client_config['model']}: size:{calculate_model_size(mdict)} MB | num params:{sum(mdict[key].nelement() for key in mdict) / 1e6} M")
        except:
            pass

    def summary_result(self):
        raise NotImplementedError

    # def summarize_test_loss_and_acc(self, **kwargs):
    #     """
    #         use -1 to represent the testing accuracy using the latest local model
    #     """
    #     client_indices = self.clients_dict.keys()
    #     # test stage
    #     if self.has_global_testset:
    #         # get global model performance
    #         test_loss, test_acc = access_last_added_element(self.server_side_client.test_loss_dict)[-1], access_last_added_element(self.server_side_client.test_acc_dict)[-1]
    #         if test_loss is None or test_acc is None:
    #             print('1) Global model is in its initial state. Please make sure at least one round of training is performed.')
    #             exit()
    #         self.average_test_loss_dict[-1], self.average_test_acc_dict[-1] = test_loss, test_acc
    #         if self.mode == "PFL":
    #             self.average_pfl_test_loss_dict[-1] = {}
    #             self.average_pfl_test_acc_dict[-1] = {}
    #             for weight in kwargs['weights']:
    #                 valid_clients = 0
    #                 total_loss = 0.0
    #                 total_acc = 0.0
    #                 total_samples = 0
    #                 for cid in client_indices:
    #                     client = self.clients_dict[cid]
    #                     c_pfl_test_loss, c_pfl_test_acc = access_last_added_element(client.test_pfl_loss_dict), access_last_added_element(client.test_pfl_acc_dict)
    #                     if c_pfl_test_loss is None or c_pfl_test_acc is None:
    #                         print(f' Client:{cid} model is in its initial state. We do not count this client into final testing accuracy.')
    #                         continue
    #                     loss, acc, num_samples = c_pfl_test_loss[weight][-1], c_pfl_test_acc[weight][-1], client.num_train_samples
    #                     valid_clients += 1
    #                     total_loss += loss * num_samples
    #                     total_acc += acc # use different scaling here
    #                     total_samples += num_samples
    #                 average_loss, average_acc = total_loss / total_samples, total_acc / valid_clients
    #                 self.average_pfl_test_loss_dict[-1][weight] = average_loss
    #                 self.average_pfl_test_acc_dict[-1][weight] = average_acc
    #             print(f"PFL: In summarize_test_loss_and_acc, valid_clients:{valid_clients} / total_clients:{len(self.clients_dict.keys())}")
