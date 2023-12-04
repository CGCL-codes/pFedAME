# pFedAME - Personalized Federated Learning with Adaptive Model Ensemble

## run command

```bash
python run_algo.py
```

## File Tree

```
.
├── experiments
│   ├── Cifar100_Conv2Cifar_cross_device.yaml
│   ├── Cifar100_Conv2Cifar_cross_silo.yaml
│   ├── Cifar10_Conv2Cifar_cross_device.yaml
│   └── Cifar10_Conv2Cifar_cross_silo.yaml
├── README.md
├── run_algo.py
├── src
│   ├── flbase
│   │   ├── client.py
│   │   ├── model.py
│   │   ├── models
│   │   │   ├── CNN.py
│   │   ├── server.py
│   │   ├── strategies
│   │   │   ├── FedAvg.py
│   │   │   └── pFedAME.py
│   │   └── utils.py
│   └── utils.py
└── test
    ├── FedAvg
    │   ├── run.py
    └── pFedAME
        ├── run.py
```