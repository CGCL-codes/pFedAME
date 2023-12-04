import os
from time import sleep

def cycle_generator(lst):
    """
    循环生成器函数，从给定的列表中循环生成元素，当到达列表末尾时，返回到列表开头。
    """
    i = 0
    while True:
        yield lst[i]
        i += 1
        if i == len(lst):
            i = 0

# run_cmd = "python run.py --keep_clients_model {} --partition {} --beta {} --num_shards_per_client {} --yamlfile {} --use_wandb True --device cuda:{} --seed {}"
run_cmd = "python run.py --keep_clients_model {} --partition {} --beta {} --num_shards_per_client {} --yamlfile {} --use_wandb False --device cuda:{} --seed {}"

# Config
keep_clients_model = True
partition = "noniid-label-distribution"
# betas = [0.3, 1.0]
betas = [0.3]
num_shards_per_client = 2
yamlfiles = [
    "../../experiments/Cifar10_Conv2Cifar_cross_device.yaml",
    # "../../experiments/Cifar100_Conv2Cifar_cross_device.yaml",
    # "../../experiments/Cifar10_Conv2Cifar_cross_silo.yaml",
    # "../../experiments/Cifar100_Conv2Cifar_cross_silo.yaml",
]
# seed_list = [2022,2023,2024]
seed_list = [2022]
device_list = cycle_generator([0,1,2,3])
algo_list = ["pFedAME"]

import subprocess

scripts_base = "/home/dzyao/ZZQ/pFedAME/test/{}"

# 遍历目录中的所有文件
for seed in seed_list:
    for algo in algo_list:
        for beta in betas:
            for yamlfile in yamlfiles:
                device_id = next(device_list)
                command = run_cmd.format(keep_clients_model, partition, beta, 
                                        num_shards_per_client, yamlfile, device_id, seed)
                os.chdir(scripts_base.format(algo))
                subprocess.Popen(command, shell=True)