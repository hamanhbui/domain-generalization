import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.Baseline.src.dataloaders import dataloader_factory
from algorithms.Baseline.src.models import model_factory
from torch.optim.lr_scheduler import StepLR

class Trainer_Baseline:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "/")
        self.train_loader = DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_train_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_val_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.target_test_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.model = model_factory.get_model(self.args.model)(classes = self.args.n_classes).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.learning_rate, weight_decay=.0005, momentum=.9, nesterov=False)
        self.criterion = nn.CrossEntropyLoss()

        # self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.checkpoint_name = "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name
        self.val_loss_min = np.Inf

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def train(self):
        self.model.train()
        n_class_corrected = 0
        total_classification_loss = 0
        total_samples = 0
        for epoch in range(self.args.epochs):
            for iteration, (samples, labels, domain_labels) in enumerate(self.train_loader):
                # self.lr_scheduler.step()
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                self.optimizer.zero_grad()
                
                predicted_classes = self.model(samples)
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()
                
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
                total_samples += len(samples)

                classification_loss.backward()
                self.optimizer.step()

                if iteration % self.args.step_eval == 0:
                    n_iter = epoch * len(self.train_loader) + iteration
                    self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, n_iter)
                    self.writer.add_scalar('Loss/train', total_classification_loss / total_samples, n_iter)
                    logging.info('Train set: Epoch: {} [{}/{}]\tAccuracy: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(epoch, (iteration + 1) * len(samples), len(self.train_loader.dataset),
                        n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
                    self.evaluate(n_iter)
                    
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

        val_loss = total_classification_loss / len(self.val_loader.dataset)
        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            torch.save(self.model.state_dict(), self.checkpoint_name + '.pt')

    def test(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name + '.pt'))
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