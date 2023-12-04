from ..model import Model
from .resnet18 import resnet_18_cifar as Resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models

"""
Ref: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""


class ModelWrapper(Model):
    def __init__(self, base, head, config):
        super(ModelWrapper, self).__init__(config)

        self.base = base
        # head is a matrix here not a nn.module
        self.head = head

    def forward(self, x, return_embedding):
        feature_embedding = self.base(x)
        out = torch.matmul(feature_embedding, self.head.T)
        if return_embedding:
            return feature_embedding, out
        else:
            return out


class CNNMnist(Model):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv2d(config['input_size'][0], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, config['num_classes'])

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNFashionMnist(Model):
    def __init__(self, config):
        super().__init__(config)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Conv2Cifar(Model):
    def __init__(self, config):
        super().__init__(config)
        self.normalize = config['normalize']
        self.return_embedding = config['return_embedding']
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        if self.normalize:
            self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
        self.return_feature = False

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)

        if self.return_feature == True:
            return x

        x = F.relu(self.fc1(x))
        feature_embedding = F.relu(self.fc2(x))
        if self.normalize:
            feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
            if self.prototype.requires_grad == False:
                normalized_prototype = self.prototype
            else:
                prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                normalized_prototype = torch.div(self.prototype, prototype_norm)
            logits = torch.matmul(feature_embedding, normalized_prototype.T)
            logits = self.scaling * logits
        else:
            if self.prototype is None:
                return feature_embedding
            logits = torch.matmul(feature_embedding, self.prototype.T)

        if self.return_embedding:
            return feature_embedding, logits
        return logits

class Conv2TinyImageNet(Model):
    def __init__(self, config):
        super().__init__(config)
        self.normalize = config['normalize']
        self.return_embedding = config['return_embedding']
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 13 * 13, 384)
        self.fc2 = nn.Linear(384, 192)
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        if self.normalize:
            self.scaling = torch.nn.Parameter(torch.tensor([1.0]))
        
        self.return_feature = False

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 13 * 13)

        if self.return_feature == True:
            return x

        x = F.relu(self.fc1(x))
        feature_embedding = F.relu(self.fc2(x))
        if self.normalize:
            feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
            if self.prototype.requires_grad == False:
                normalized_prototype = self.prototype
            else:
                prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                normalized_prototype = torch.div(self.prototype, prototype_norm)
            logits = torch.matmul(feature_embedding, normalized_prototype.T)
            logits = self.scaling * logits
        else:
            if self.prototype is None:
                return feature_embedding
            logits = torch.matmul(feature_embedding, self.prototype.T)

        if self.return_embedding:
            return feature_embedding, logits
        return logits

"""
MobileNet Cifar V1
Ref: https://github.com/jhoon-oh/FedBABU/blob/master/models/Nets.py
"""


class MobileNetV1Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1):
        super(MobileNetV1Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetV1Cifar(Model):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, config):
        num_cls = config['num_classes']
        self.normalize = config['normalize']
        self.return_embedding = config['return_embedding']
        super().__init__(config)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        temp = nn.Linear(1024, num_cls, bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(MobileNetV1Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        feature_embedding = out.view(out.size(0), -1)
        # logits = self.linear(feature_embedding)
        if self.return_embedding:
            return feature_embedding

        if self.normalize:
            feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
            logits = torch.matmul(feature_embedding, normalized_prototype.T)
        else:
            logits = torch.matmul(feature_embedding, self.prototype.T)

        return logits

