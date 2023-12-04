from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from ..models.CNN import *
from ..models.RNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch
import asyncio


class FedAvgClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self._initialize_model()
        self.initial_model_sd = None

    def _initialize_model(self):
        # parse the model from config file
        self.model = eval(self.client_config['model'])(self.client_config).to(self.device)
        self.criterion = self.criterion.to(self.device)

    def training(self, round, num_epochs):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
        # train mode
        self.model.train()
        # tracking stats
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")
        optimizer = setup_optimizer(self.model, self.client_config, round)

        # training starts
        # print(f'---- Train Client:{self.cid}')
        # print(self.cid)
        # print('delete line 62-63 FedAvg')
        # print('start W', self.model.state_dict()['prototype'][0, :3])
        self.initial_model_sd = deepcopy(self.model.state_dict())
        # gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        self.model = nn.DataParallel(self.model).cuda()
        
        # epoches_iter = tqdm(range(num_epochs))
        epoches_iter = range(num_epochs)
        for i in epoches_iter:
            # print('start W', self.model.state_dict()['prototype'][0, :3])
            # epoches_iter.set_description("Processing Client %s" % self.cid)
            epoch_loss, correct = 0.0, 0
            for j, (x, y) in enumerate(self.trainloader):
                # forward pass
                # x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                yhat = self.model.forward(x)
                # yhat = model.forward(x)
                loss = self.criterion(yhat, y)
                # backward pass
                # model.zero_grad safer and memory-efficient
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()
                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)
        
        self.model = self.model.module.to(self.device)
        self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq

    def upload(self):
        return self.new_state_dict

    def testing(self, round, testloader=None, record_result=True):
        self.model.eval()

        if record_result:
            self.test_model = deepcopy(self.model)
        else:
            self.test_model = deepcopy(self.model)
            self.test_model.load_state_dict(self.initial_model_sd)

        if self.initial_model_sd is None: # on server
            pass
        else: # on client
            if 'testglo' in list(self.client_config.keys()) and self.client_config['testglo']:
                self.test_model.load_state_dict(self.initial_model_sd)
        
        if testloader is None:
            testloader = self.testloader
        test_count_per_class = Counter(testloader.dataset.targets.numpy())
        all_classes_sorted = sorted(test_count_per_class.keys())
        test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in all_classes_sorted])
        num_classes = len(all_classes_sorted)
        test_correct_per_class = torch.tensor([0] * num_classes)

        weight_per_class_dict = {'uniform': torch.tensor([1.0] * num_classes),
                                 'validclass': torch.tensor([0.0] * num_classes),
                                 'labeldist': torch.tensor([0.0] * num_classes)}
        for cls in self.label_dist.keys():
            weight_per_class_dict['labeldist'][cls] = self.label_dist[cls]
            weight_per_class_dict['validclass'][cls] = 1.0
        # start testing
        with torch.no_grad():
            for (x,y) in testloader:
                # forward pass
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                yhat = self.test_model.forward(x)
                # stats
                predicted = yhat.data.max(1)[1]
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    test_correct_per_class[cls] += ((predicted == y) * (y == cls)).sum().item()
        acc_by_critertia_dict = {}
        for k in weight_per_class_dict.keys():
            acc_by_critertia_dict[k] = (((weight_per_class_dict[k] * test_correct_per_class).sum()) /
                                        ((weight_per_class_dict[k] * test_count_per_class).sum())).item()
        
        if record_result:
            print(f"{self.cid} after train: {acc_by_critertia_dict}")
            self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'weight_per_class': weight_per_class_dict}
        else:
            print(f"{self.cid} before train: {acc_by_critertia_dict}")


class FedAvgServer(Server):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, **kwargs)
        self.summary_setup()
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        # make sure the starting point is correct
        self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
        self.exclude_layer_keys = set()
        for key in self.server_model_state_dict:
            for ekey in exclude:
                if ekey in key:
                    self.exclude_layer_keys.add(key)
        if len(self.exclude_layer_keys) > 0:
            print(f"{self.server_config['strategy']}Server: the following keys will not be aggregate:\n ", self.exclude_layer_keys)
        freeze_layers = []
        for param in self.server_side_client.model.named_parameters():
            if param[1].requires_grad == False:
                freeze_layers.append(param[0])
        if len(freeze_layers) > 0:
            print("{self.server_config['strategy']}Server: the following layers will not be updated:", freeze_layers)

    def aggregate(self, client_uploads, round):
        server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        num_participants = len(client_uploads)
        update_direction_state_dict = None
        exclude_layer_keys = self.exclude_layer_keys
        with torch.no_grad():
            for idx, client_state_dict in enumerate(client_uploads):
                client_update = linear_combination_state_dict(client_state_dict,
                                                              self.server_model_state_dict,
                                                              1.0,
                                                              -1.0,
                                                              exclude=exclude_layer_keys
                                                              )
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0,
                                                                                exclude=exclude_layer_keys
                                                                                )
            # new global model
            self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
                                                                         update_direction_state_dict,
                                                                         1.0,
                                                                         server_lr / num_participants,
                                                                         exclude=exclude_layer_keys
                                                                         )
            print(' ', self.server_config['strategy'])

    def testing(self, round, active_only=True, **kwargs):
        """
        active_only: only compute statiscs with to the active clients only
        """
        # get the latest global model
        self.server_side_client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

        # test the performance for global models
        self.server_side_client.testing(round, testloader=None)  # use global testdataset
        print('server global model correct', torch.sum(self.server_side_client.test_acc_dict[round]['correct_per_class']).item())
        # test the performance for local models (potentiallt only for active local clients)
        client_indices = self.clients_dict.keys()
        if active_only:
            client_indices = self.active_clients_indicies
        for cid in client_indices:
            client = self.clients_dict[cid]
            # test local model on the global testset
            # client.testing(round, self.server_side_client.testloader, record_result=False)
            client.testing(round, self.server_side_client.testloader)

    def collect_stats(self, stage, round, active_only, **kwargs):
        """
            No actual training and testing is performed. Just collect stats.
            stage: str;
                {"train", "test"}
            active_only: bool;
                True: compute stats on active clients only
                False: compute stats on all clients
        """
        # get client_indices
        client_indices = self.clients_dict.keys()
        if active_only:
            client_indices = self.active_clients_indicies
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        if stage == 'train':
            for cid in client_indices:
                client = self.clients_dict[cid]
                # client.train_loss_dict[round] is a list compose the training loss per end of each epoch
                loss, acc, num_samples = client.train_loss_dict[round][-1], client.train_acc_dict[round][-1], client.num_train_samples
                total_loss += loss * num_samples
                total_acc += acc * num_samples
                total_samples += num_samples
            average_loss, average_acc = total_loss / total_samples, total_acc / total_samples
            self.average_train_loss_dict[round] = average_loss
            self.average_train_acc_dict[round] = average_acc
        else:
            # test stage
            # get global model performance
            self.gfl_test_acc_dict[round] = self.server_side_client.test_acc_dict[round]
            acc_criteria = self.server_side_client.test_acc_dict[round]['acc_by_criteria'].keys()
            # get local model average performance
            self.average_pfl_test_acc_dict[round] = {key: 0.0 for key in acc_criteria}
            for cid in client_indices:
                client = self.clients_dict[cid]
                acc_by_criteria_dict = client.test_acc_dict[round]['acc_by_criteria']
                for key in acc_criteria:
                    self.average_pfl_test_acc_dict[round][key] += acc_by_criteria_dict[key]

            num_participants = len(client_indices)
            for key in acc_criteria:
                self.average_pfl_test_acc_dict[round][key] /= num_participants

    def run(self, **kwargs):
        self.run_id = kwargs["run_id"]
        if self.server_config['use_tqdm']:
            round_iterator = tqdm(range(self.rounds + 1, self.server_config['num_rounds'] + 1), desc="Round Progress")
        else:
            round_iterator = range(self.rounds + 1, self.server_config['num_rounds'] + 1)
        # round index begin with 1
        for r in round_iterator:
            selected_indices = self.select_clients(self.server_config['participate_ratio'])
            if self.server_config['drop_ratio'] > 0:
                # mimic the stragler issues; simply drop them
                self.active_clients_indicies = np.random.choice(selected_indices, int(
                    len(selected_indices) * (1 - self.server_config['drop_ratio'])), replace=False)
            else:
                self.active_clients_indicies = selected_indices
            # active clients download weights from the server
            tqdm.write(f"Round:{r} - Active clients:{self.active_clients_indicies}:")
            from datetime import datetime
            timestamp = datetime.now().strftime("%m.%d  %H:%M")
            tqdm.write(f"Starting Time: {timestamp}")
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

            # clients perform local training
            train_start = time.time()
            client_uploads = []
            
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.training(r, client.client_config['num_epochs'])
                client_uploads.append(client.upload())
            
            train_time = time.time() - train_start
            print(f" Training time:{train_time:.3f} seconds")
            # collect training stats
            # average train loss and acc over active clients, where each client uses the latest local models
            self.collect_stats(stage="train", round=r, active_only=True)

            # get new server model
            self.aggregate(client_uploads, round=r)
            # agg_time = time.time() - agg_start
            # print(f" Aggregation time:{agg_time:.3f} seconds")
            # collect testing stats
            if (r - 1) % self.server_config['test_every'] == 0:
                test_start = time.time()
                self.testing(round=r, active_only=True)
                test_time = time.time() - test_start
                print(f" Testing time:{test_time:.3f} seconds")
                self.collect_stats(stage="test", round=r, active_only=True)
                print(" avg_test_acc:", self.gfl_test_acc_dict[r]['acc_by_criteria'])
                print(" pfl_avg_test_acc:", self.average_pfl_test_acc_dict[r])
                if len(self.gfl_test_acc_dict) >= 2:
                    current_key = r
                    if self.gfl_test_acc_dict[current_key]['acc_by_criteria']['uniform'] > best_test_acc:
                        best_test_acc = self.gfl_test_acc_dict[current_key]['acc_by_criteria']['uniform']
                        self.server_model_state_dict_best_so_far = deepcopy(self.server_model_state_dict)
                        tqdm.write(f" Best test accuracy:{float(best_test_acc):5.3f}. Best server model is updatded and saved at {kwargs['filename']}!")
                        if 'filename' in kwargs:
                            torch.save(self.server_model_state_dict_best_so_far, 
                                        f"{kwargs['filename']}.{self.run_id}")
                else:
                    best_test_acc = self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']

            # wandb monitoring
            if kwargs['use_wandb']:
                stats = {"avg_train_loss": self.average_train_loss_dict[r],
                         "avg_train_acc": self.average_train_acc_dict[r],
                         "gfl_test_acc_uniform": self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
                         }

                for criteria in self.average_pfl_test_acc_dict[r].keys():
                    stats[f'pfl_test_acc_{criteria}'] = self.average_pfl_test_acc_dict[r][criteria]

                wandb.log(stats)