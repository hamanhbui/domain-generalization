import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.AGG.src.dataloaders import dataloader_factory
from algorithms.AGG.src.models import model_factory
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR

class EarlyStopping:
    def __init__(self, checkpoint_name, lr_scheduler, patiences=[], delta=0):
        self.checkpoint_name = checkpoint_name
        self.lr_scheduler = lr_scheduler
        self.ignore_times = len(patiences)
        self.patience_idx = 0
        self.patiences = patiences
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patiences[self.patience_idx]:
                model.load_state_dict(torch.load(self.checkpoint_name))
                if self.patience_idx < self.ignore_times - 1:
                    # self.lr_scheduler.step()
                    self.patience_idx += 1
                    self.counter = 0
                else:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.checkpoint_name)
        self.val_loss_min = val_loss

class Trainer_AGG:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "/")
        self.train_loaders = [DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[0]], domain_label = 0), batch_size = self.args.batch_size, shuffle = True),
        DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[1]], domain_label = 1), batch_size = self.args.batch_size, shuffle = True),
        DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = [self.args.src_train_meta_filenames[2]], domain_label = 2), batch_size = self.args.batch_size, shuffle = True)]
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_val_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.target_test_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.model = model_factory.get_model(self.args.model)(classes = self.args.n_classes, pretrained=False).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.learning_rate, weight_decay=1e-4, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_name = "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + '.pt'
        self.scheduler = StepLR(self.optimizer, step_size=3001, gamma=0.1)
        # self.early_stopping = EarlyStopping(checkpoint_name = "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + '.pt', 
        #     lr_scheduler = StepLR(self.optimizer, step_size=3001, gamma=0.1), patiences=[10005])
        self.load_state_dict(self.model)

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def load_state_dict(self, nn):
        
        try:
            tmp = torch.load("https://download.pytorch.org/models/resnet18-5c106cde.pth")
            if 'state' in tmp.keys():
                pretrained_dict = tmp['state']
            else:
                pretrained_dict = tmp
        except:
            pretrained_dict = torch.load("pretrained_models/resnet18-5c106cde.pth")
            
        model_dict = nn.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}

        print('model dict keys:', len(model_dict.keys()), 'pretrained keys:', len(pretrained_dict.keys()))
        print('model dict keys:', model_dict.keys(), 'pretrained keys:', pretrained_dict.keys())
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        nn.load_state_dict(model_dict)

    def train(self):
        self.model.train()
        n_class_corrected = 0
        total_classification_loss = 0
        total_samples = 0
        itr = 0
        epoch = 0
        self.train_iter_loaders = []
        for train_loader in self.train_loaders:
            self.train_iter_loaders.append(iter(train_loader))

        while True:
            epoch += 1
            self.scheduler.step(epoch = itr)
            itr += 1
            if itr > 3001:
                return
            
            total_domain_classification_loss = 0
            for idx in range(len(self.train_iter_loaders)):
                if (itr % len(self.train_iter_loaders[idx])) == 0:
                    self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
                train_loader = self.train_iter_loaders[idx]

                samples, labels, domain_labels = train_loader.next()
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                
                predicted_classes = self.model(samples)
                classification_loss = self.criterion(predicted_classes, labels)
                total_domain_classification_loss += classification_loss
                total_classification_loss += classification_loss.item()
                    
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
                total_samples += len(samples)

            self.optimizer.zero_grad()
            total_domain_classification_loss.backward()
            self.optimizer.step()

            if itr % 100 == 0 and itr is not 0:
                # n_iter = epoch * len(self.train_loader) + iteration
                # self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, n_iter)
                # self.writer.add_scalar('Loss/train', total_classification_loss / total_samples, n_iter)
                # logging.info('Train set: Epoch: {} [{}/{}]\tAccuracy: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(epoch, (iteration + 1) * len(samples), len(self.train_loader.dataset),
                #     n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
                self.evaluate(itr)
                self.test()
                # if self.early_stopping.early_stop:
                    # return
                n_class_corrected = 0
                total_classification_loss = 0
                total_samples = 0
                self.model.train()
            
            n_class_corrected = 0
            total_classification_loss = 0
            total_samples = 0
    
    def evaluate(self, n_iter):
        self.model.eval()
        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.val_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                predicted_classes = self.model(samples)
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        self.writer.add_scalar('Accuracy/validate', 100. * n_class_corrected / len(self.val_loader.dataset), n_iter)
        self.writer.add_scalar('Loss/validate', total_classification_loss / len(self.val_loader.dataset), n_iter)
        logging.info('Val set: Accuracy: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(n_class_corrected, len(self.val_loader.dataset),
            100. * n_class_corrected / len(self.val_loader.dataset), total_classification_loss / len(self.val_loader.dataset)))
        # self.early_stopping(total_classification_loss / len(self.val_loader.dataset), self.model)            
            
    def test(self): 
        self.model.eval()
        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                predicted_classes = self.model(samples)
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.0f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))