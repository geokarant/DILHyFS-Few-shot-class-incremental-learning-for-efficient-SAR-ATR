import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import Cross_Attention,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy, PairedDataset
from fvcore.nn import FlopCountAnalysis,flop_count_table
from torchprofile import profile_macs
from utils.losses import FocalLoss,CenterLoss
import sys        
import copy

# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args,True)
        self. batch_size = args["batch_size"]
        self. init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.cov_mats, self.base_cov_mats = [], []
        self.proto_list = []
        self.ridge = 0
        self.current_class = 0

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):       
        model = model.eval()
        embedding_list = []
        label_list = []
        cur_proto_list = []

        # Iterate through both the ViT and CNN loaders in parallel
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        Y = target2onehot(label_list, self.args["nb_classes"])
        if self.args["use_RP"] == True:
            Features_h = F.relu(embedding_list @ self.W_rand.cpu())
        else:
            Features_h = embedding_list
        print(Features_h.shape)
        if self.args['eq_prot'] == True: 
            class_counts = torch.bincount(label_list.to(torch.int64))       
            inv_class_frequencies = 1.0 / class_counts
            for cls in range(self.current_class ,len(class_counts)):
                cls_mask = (label_list == cls)
                Features_h_cls = Features_h[cls_mask]
                Y_cls = Y[cls_mask]
                weight = inv_class_frequencies[cls]
                self.Q[:, cls] += weight * (Features_h_cls.T @ Y_cls[:, cls])
                self.current_class += 1
        else:
            self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h
        if self.args["lda"] == True:
            if self._cur_task == 0:
                print(self._cur_task, "Task : Calculating λ")
                self.ridge = self.optimise_ridge_parameter(Features_h, Y)
            print(f"λ = {self.ridge}")
            Wo = torch.linalg.solve(self.G + self.ridge*torch.eye(self.G.size(dim=0)), self.Q).T # better nmerical stability than .invv
        else:
            Wo = self.Q.T
        self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0],:].to(self._device)
        return model
    
    def setup_RP(self):
        if self.args["use_RP"] == True :
            M = self.args['M']
            self._network.RP_dim = M
            self.W_rand = torch.randn(self._network.fc.in_features, M).to(self._device)
            self._network.W_rand = self.W_rand
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(self._device)).requires_grad_(False) # num classes in task x M
            self.Q = torch.zeros(M, self.args["nb_classes"])
            self.G = torch.zeros(M, M)
        else:
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, self._network.feature_dim).to(self._device)).requires_grad_(False) # num classes in task x M
            self.Q = torch.zeros(self._network.feature_dim, self.args["nb_classes"])
            self.G = torch.zeros(self._network.feature_dim, self._network.feature_dim)

    def optimise_ridge_parameter(self, Features_h, Y_h):
        if self.args['random_permutation'] == True:
            indices = np.random.permutation(Features_h.shape[0])
            Features= Features_h[indices]
            Y = Y_h[indices]
        else:
            Features = Features_h
            Y = Y_h    
        ridges = 10.0 ** np.arange(-4, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        losses = torch.tensor(losses)
        ridge = ridges[np.argmin(losses.detach().numpy())]
        return ridge
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._network._cur_task = self._cur_task
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
       
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train" )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        #print(test_dataset)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test" )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if  self.args['finetune'] == True:
            if self._cur_task == 0:
                self._network.enable_parameters()
                # show total parameters and trainable parameters
                total_params = sum(p.numel() for p in self._network.parameters())
                print(f'{total_params:,} total parameters.')
                total_trainable_params = sum(
                    p.numel() for p in self._network.parameters() if p.requires_grad)
                print(f'{total_trainable_params:,} total trainable parameters.')
                if self.args['optimizer'] == 'sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
                elif self.args['optimizer'] == 'adam':
                    optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                pass
        #self.savemodel()
        if self._cur_task == 0 and self.args["use_RP"]:
            self.setup_RP()
        self.replace_fc(train_loader_for_protonet, self._network, self.args)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        function_loss = FocalLoss(self.args['gamma'])
        center_loss = CenterLoss(self._network.fc.weight.shape[0], self._network.feature_dim,self._device).to(self._device)
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                features= self._network.extract_vector(inputs)
                logits = self._network(inputs)["logits"]
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100)
                if self.args["loss"] == 'ce':
                    loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)
                elif self.args["loss"] == 'focal':
                    loss = function_loss(logits,targets)
                elif self.args["loss"] == 'focal+center':
                    loss = function_loss(logits,targets)
                    loss+= self.args["alpha"] *center_loss(features, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        for name, param in self._network.named_parameters():
            param.requires_grad = False  # Freeze other parameters
        logging.info(info)
    def savemodel(self):
        del self._network.fc
        torch.save(self._network.state_dict(), '')
        sys.exit()