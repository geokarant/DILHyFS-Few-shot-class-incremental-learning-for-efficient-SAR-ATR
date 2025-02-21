import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import csv
import numpy as np
import time
import random
import itertools
def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    torch.cuda.synchronize()
    total_time = 0
    args["device"] = device
    start_time = time.time()

    for seed in seed_list:
        args["seed"]= seed
        args["device"]= device
        if args["early_fusion"] != False:
            acc_task, avg_acc = _train_early_fusion(args)
        else:
            acc_task, avg_acc = _train(args)
    torch.cuda.synchronize()  # Ensure the forward pass is completed before measuring time
    end_time = time.time()
    total_time += (end_time - start_time)
    logging.info(f"total training time: {total_time} seconds")

def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    cnn_curve_hm = []
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))

        model.incremental_train(data_manager)
        
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        #model.savemodel()
       
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve_hm.append(cnn_accy["hm"])
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {:.2f} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
    return cnn_curve["top1"], np.around(sum(cnn_curve["top1"])/len(cnn_curve["top1"]), decimals=2)

def _train_early_fusion(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type_vit"]
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    args_vit = copy.deepcopy(args)
    args_vit["backbone_type"] = args["backbone_type_vit"]
    data_manager_vit = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args_vit,
    )
    args["nb_classes"] = data_manager_vit.nb_classes # update args
    args["nb_tasks"] = data_manager_vit.nb_tasks

    args_cnn = copy.deepcopy(args)
    args_cnn["backbone_type"] = args["backbone_type_cnn"]
    data_manager_cnn = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args_cnn,
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    cnn_curve_hm = []

    for task in range(data_manager_vit.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))

        model.incremental_train(data_manager_vit, data_manager_cnn)
        cnn_accy, nme_accy = model.eval_task_early_fusion()
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)
        cnn_curve_hm.append(cnn_accy["hm"])

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {:.2f} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
    return cnn_curve["top1"], np.around(sum(cnn_curve["top1"])/len(cnn_curve["top1"]), decimals=2)
        
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus

def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))