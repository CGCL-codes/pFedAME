from torch import nn
import torch
from ..utils import autoassign, calculate_model_size, calculate_flops
from tqdm import tqdm, trange
from .utils import setup_optimizer


class Model(nn.Module):
    """For classification problem"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_params(self):
        return self.state_dict()

    def get_gradients(self, dataloader):
        raise NotImplementedError

    def set_params(self, model_state_dict, exclude_keys=set()):
        """
            Reference: Be careful with the state_dict[key].
            https://discuss.pytorch.org/t/how-to-copy-a-modified-state-dict-into-a-models-state-dict/64828/4.
        """
        with torch.no_grad():
            for key in model_state_dict.keys():
                if key not in exclude_keys:
                    self.state_dict()[key].copy_(model_state_dict[key])

    # def model_training(self, trainloader, num_epochs, round):
    #     if trainloader is None:
    #         raise ValueError("No trainloader is provided!")
    #     self.train()
    #     loss_seq = []
    #     acc_seq = []
    #     total_flops = 0.0
    #     optimizer = setup_optimizer(self, self.config, round)
    #     if self.config['use_tqdm']:
    #         tqdm_epoch_bar = tqdm(range(num_epochs), desc="Epoch Progress")
    #         tqdm_batch_bar = tqdm(total=len(trainloader),
    #                               desc="Batch progress")
    #         epoch_iterator = tqdm_epoch_bar
    #     else:
    #         epoch_iterator = range(num_epochs)
    #     for i in epoch_iterator:
    #         epoch_loss, correct = 0.0, 0
    #         if self.config['use_tqdm']:
    #             tqdm_batch_bar.reset()
    #         for _, (x, y) in enumerate(trainloader):
    #             # forward pass
    #             x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
    #             yhat = self.forward(x)
    #             loss = self.criterion(yhat, y)

    #             # backward pass
    #             # model.zero_grad safer and memory-efficient
    #             self.zero_grad(set_to_none=True)
    #             loss.backward()
    #             optimizer.step()
    #             # stats
    #             predicted = yhat.data.max(1)[1]
    #             correct += predicted.eq(y.data).sum().item()
    #             epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
    #             if self.config['use_tqdm']:
    #                 tqdm_batch_bar.update()

    #         epoch_loss /= len(trainloader.dataset)
    #         epoch_accuracy = correct / len(trainloader.dataset)
    #         loss_seq.append(epoch_loss)
    #         acc_seq.append(epoch_accuracy)
    #         # tqdm.write(f" Epoch{i:4d}: loss:{epoch_loss:3.4e} | acc:{epoch_accuracy:3.4e}")
    #         total_flops += self.flops * len(trainloader.dataset)
    #     return self.state_dict(), total_flops, loss_seq, acc_seq

    # def testing(self, testloader, weight_dict=None):
    #     if testloader is None:
    #         print("No testsets are created for clients. One should maintain a global test dataset on the server side.")
    #         raise ValueError("No testloader is provided!")
    #     self.eval()
    #     epoch_loss, correct = 0.0, 0
    #     normalization_const = 0
    #     for i, (x, y) in enumerate(testloader):
    #         with torch.no_grad():
    #             # forward pass
    #             x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
    #             yhat = self.forward(x)
    #             loss = self.criterion(yhat, y)
    #             # stats
    #             predicted = yhat.data.max(1)[1]
    #             if weight_dict is None:
    #                 correct += predicted.eq(y.data).sum().item()
    #                 normalization_const += len(y)
    #             else:
    #                 # loss is less meaningful in this case.
    #                 batch_weights = torch.tensor([weight_dict[i.item()] if i.item() in weight_dict else 0.0 for i in y]).to(self.device)
    #                 correct += torch.sum(predicted.eq(y.data) * batch_weights).item()
    #                 normalization_const += torch.sum(batch_weights).item()
    #             epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
    #     epoch_loss /= len(testloader.dataset)
    #     epoch_accuracy = correct / normalization_const
    #     return epoch_loss, epoch_accuracy
