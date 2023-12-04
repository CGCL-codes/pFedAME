from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from ..models.CNN import *
from ..models.MLP import *
from ..models.RNN import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch
import torch.nn.functional as F
from .FedAvg import FedAvgClient, FedAvgServer

class ensembled_model(Model):
    def __init__(self, model_list):
        super().__init__({})
        self.models = nn.ModuleList(deepcopy(model_list))
        self.weights = nn.ParameterList([
            torch.nn.Parameter(torch.tensor(1.0))
            for _ in range(len(model_list))
        ])
        for module in self.models:
            for param in module.parameters():
                param.requires_grad = False  # 将参数的requires_grad设置为False

    def forward(self, x):
        logits_list = []
        for model, weight in zip(self.models, self.weights):
            logits_list.append(model(x) * weight)
        # Apply softmax to the weighted logits
        logits_sum = torch.stack(logits_list).sum(dim=0)
        return torch.softmax(logits_sum, dim=1)


class pFedAMEClient(FedAvgClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self._initialize_model()
        self.state_model = deepcopy(self.model)
    
        model_list = []
        if self.client_config['ensemble_global']:
            model_list.append(self.model)
        if self.client_config['ensemble_local']: # 初始化，这个 ensemble_model 不会被用到
            model_list.append(self.model)
        if self.client_config['s_mu'] > 0:
            model_list.append(self.model)
        self.ensemble_model = ensembled_model(model_list)

    
    def _add_prox_term(self):
        for _param, _init_param in zip(
            self.model.module.parameters(), self.global_model.parameters()
        ):
            if _param.grad is not None:
                _param.grad.data.add_(
                    (_param.data - _init_param.data.to(_param.device)) * self.client_config['mu'] 
                )
    
    def add_regularization_term(self):
        self._add_prox_term()
    
    def _get_mu(self):
        if self.client_config['fix_mu'] == True: 
            return self.client_config['s_mu']
        
        total_local_round = self.client_config['num_rounds'] * self.client_config['participate_ratio'] 
        mu = self.client_config['s_mu'] * self.num_rounds_particiapted / total_local_round
        return mu if mu <= 1.0 else 1.0
    
    def weighted_aggreagte(self):
        _mu = self._get_mu()
        _ans_model = deepcopy(self.model)
        weight_keys = list(_ans_model.state_dict().keys())
        _ans_model_sd = OrderedDict()

        model_sd = self.model.state_dict()
        per_model_sd = self.state_model.state_dict()

        for key in weight_keys:
            _ans_model_sd[key] =  (1-_mu) * model_sd[key] + _mu * per_model_sd[key]

        _ans_model.load_state_dict(_ans_model_sd)
        return _ans_model
    
    def divergence(self, p, q):
        # 将分布转化为概率
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        
        loss = (p-q).abs().sum()/q.shape[0]
        
        return loss
    
    
    def ensemble_distillation(self, round):
        global_model = deepcopy(self.global_model)
        local_model = deepcopy(self.model)
        state_model = deepcopy(self.state_model)

        model_list = []
        if self.client_config['ensemble_global']:
            model_list.append(global_model)
        if self.client_config['ensemble_local']: 
            model_list.append(local_model)
        if self.client_config['s_mu'] > 0:
            model_list.append(state_model)
        self.ensemble_model = ensembled_model(model_list)

        # train ensemble coefficient
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.ensemble_model.parameters()), lr=self.client_config['adaptive_lr'])
        for i in range(self.client_config['adaptive_round']):
            self.ensemble_model.train()
            for j, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.ensemble_model.forward(x)

                loss = self.criterion(yhat, y)

                self.ensemble_model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.ensemble_model.parameters()), max_norm=10)
                optimizer.step()
        weight = ["{:.5f}".format(w.item()) for w in self.ensemble_model.weights.state_dict().values()]
        
        if self.client_config['distillation_model'] == "glo":
            finetune_model = deepcopy(global_model)
        if self.client_config['distillation_model'] == "loc":
            finetune_model = deepcopy(local_model)
        if self.client_config['distillation_model'] == "per":
            finetune_model = deepcopy(state_model)
            
        coefficient = round / self.client_config['num_rounds']
        alpha = self.client_config['hard_label_alpha'] 
        beta = self.client_config['soft_label_beta'] # * coefficient
        lr = self.client_config['distillation_lr'] 
        
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=lr)
        self.ensemble_model.eval()
        for i in range(self.client_config['distillation_round']):
            finetune_model.train()
            epoch_loss, epoch_cs_loss, epoch_kl_loss, correct = 0.0, 0.0, 0.0, 0.0
            for j, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = finetune_model.forward(x)
                yensemble = self.ensemble_model.forward(x)

                cs_loss = self.criterion(yhat, y)
                kl_loss = self.divergence(yensemble, yhat)
                loss = alpha * cs_loss + beta * kl_loss

                finetune_model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, finetune_model.parameters()), max_norm=10)
                optimizer.step()

                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
                epoch_cs_loss += cs_loss.item() * x.shape[0]  # rescale to bacthsize
                epoch_kl_loss += kl_loss.item() * x.shape[0]  # rescale to bacthsize

            epoch_loss /= len(self.trainloader.dataset)
            epoch_cs_loss /= len(self.trainloader.dataset)
            epoch_kl_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
        
        return finetune_model
    

    def training(self, round, num_epochs):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
        # train mode
        self.global_model = deepcopy(self.model)
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")

        # train global model
        self.initial_model_sd = deepcopy(self.model.state_dict())
        self.model = nn.DataParallel(self.model).cuda()
        optimizer = setup_optimizer(self.model, self.client_config, round)
        for i in range(num_epochs):
            self.model.train()
            epoch_loss, correct = 0.0, 0.0
            for j, (x, y) in enumerate(self.trainloader):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                yhat = self.model.forward(x)
                loss = self.criterion(yhat, y)
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.add_regularization_term()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()
                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize

            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)

        ## APFL aggregate
        self.model = self.model.module.to(self.device)
        self.local_model = deepcopy(self.model)
        if self.num_rounds_particiapted == 1 and self.client_config['s_mu'] == 0:
            self.state_model = deepcopy(self.local_model)
        else:
            self.state_model = self.weighted_aggreagte()

        if self.client_config['distillation_round'] > 0:
            upload_model = self.ensemble_distillation(round)
            self.new_state_dict = upload_model.state_dict()
        else:
            self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq
    
    def testing(self, round, testloader=None):
        self.model.eval()
        if self.initial_model_sd is None:
            self.test_model = deepcopy(self.model)
        else:
            initial_model = deepcopy(self.model)
            initial_model.load_state_dict(self.initial_model_sd)
            # self.test_model = deepcopy(initial_model)
            # self.test_model = deepcopy(self.model)
            if self.client_config['distillation_round'] > 0:
                self.test_model = self.ensemble_model
            else:
                model_list = []
                if self.client_config['ensemble_global']:
                    model_list.append(initial_model)
                if self.client_config['ensemble_local']: 
                    model_list.append(self.model)
                if self.client_config['s_mu'] > 0:
                    model_list.append(self.state_model)
                self.ensemble_model = ensembled_model(model_list)
                self.test_model = self.ensemble_model
                
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

        self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'weight_per_class': weight_per_class_dict}


class pFedAMEServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        # initialize server state
        model_template = self.clients_dict[0].model
        self.server_state_sd = {}; model_sd = model_template.state_dict()
        for key in model_sd.keys():
            self.server_state_sd[key] = torch.zeros_like(model_sd[key])
        self.fedavg_model_sd = self.server_model_state_dict
    
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
            # momentum way
            _alpha = self.server_config['alpha'] 
            self.server_state_sd = linear_combination_state_dict(self.server_model_state_dict, 
                                                        update_direction_state_dict,
                                                        1.0,
                                                        - _alpha / num_participants,
                                                        exclude=exclude_layer_keys
                                                        )
            self.fedavg_model_sd = linear_combination_state_dict(self.fedavg_model_sd,
                                                            update_direction_state_dict,
                                                            1.0,
                                                            server_lr / num_participants,
                                                            exclude=exclude_layer_keys
                                                            )
            self.server_model_state_dict = linear_combination_state_dict(self.fedavg_model_sd,
                                                            self.server_state_sd,
                                                            1.0,
                                                            - (1 / _alpha),
                                                            exclude=exclude_layer_keys
                                                            )

            # if only fedavg, comment above code and uncomment following 
            # self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
            #                                                              update_direction_state_dict,
            #                                                              1.0,
            #                                                              server_lr / num_participants,
            #                                                              exclude=exclude_layer_keys
            #                                                              )

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
    
        for cid in self.clients_dict.keys():
            client = self.clients_dict[cid]
            weight = ["{:.5f}".format(w.item()) for w in client.ensemble_model.weights.state_dict().values()]
            print(f"Client{cid} Ensemble Coefficient at final round: {weight}")