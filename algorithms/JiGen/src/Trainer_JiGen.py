import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.JiGen.src.dataloaders import dataloader_factory
from algorithms.JiGen.src.models import model_factory
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
                    self.lr_scheduler.step()
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

class Trainer_JiGen:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "/")
        self.train_loader = DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_train_meta_filenames, n_jig_classes = self.args.n_jig_classes), batch_size = self.args.batch_size, shuffle = True)
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_val_meta_filenames, n_jig_classes = self.args.n_jig_classes), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.target_test_meta_filenames, n_jig_classes = self.args.n_jig_classes), batch_size = self.args.batch_size, shuffle = True)
        self.model = model_factory.get_model(self.args.model)(jigsaw_classes=self.args.n_jig_classes + 1, classes = self.args.n_classes).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.args.learning_rate, weight_decay=.0005, momentum=.9, nesterov=False)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(checkpoint_name = "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + '.pt', 
            lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5), patiences=[10, 5])

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def train(self):
        self.model.train()
        n_class_corrected = 0
        total_classification_loss = 0
        total_jigen_loss = 0
        total_sum_loss = 0
        total_samples = 0
        for epoch in range(self.args.epochs):
            for iteration, (samples, labels, domain_labels, jigen_labels) in enumerate(self.train_loader):
                samples, labels, domain_labels, jigen_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device), jigen_labels.to(self.device)
                self.optimizer.zero_grad()
                
                predicted_jigen_classes, predicted_classes = self.model(samples)
                jigen_loss = self.criterion(predicted_jigen_classes, jigen_labels)
                classification_loss = self.criterion(predicted_classes[jigen_labels == 0], labels[jigen_labels == 0])

                total_classification_loss += classification_loss.item()
                total_jigen_loss += jigen_loss.item()
                
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
                total_samples += len(samples)

                sum_loss = classification_loss + jigen_loss * self.args.lambda_jigen
                total_sum_loss = sum_loss.item()

                sum_loss.backward()
                self.optimizer.step()

                if iteration % self.args.step_eval == 0:
                    n_iter = epoch * len(self.train_loader) + iteration
                    self.writer.add_scalar('Accuracy/train', 100. * n_class_corrected / total_samples, n_iter)
                    self.writer.add_scalar('Loss/classification_loss', total_classification_loss / total_samples, n_iter)
                    self.writer.add_scalar('Loss/jigen_loss', total_jigen_loss / total_samples, n_iter)
                    self.writer.add_scalar('Loss/sum_loss', total_sum_loss / total_samples, n_iter)
                    logging.info('Train set: Epoch: {} [{}/{}]\tAccuracy: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(epoch, (iteration + 1) * len(samples), len(self.train_loader.dataset),
                        n_class_corrected, total_samples, 100. * n_class_corrected / total_samples, total_classification_loss / total_samples))
                    self.evaluate(n_iter)
                    if self.early_stopping.early_stop:
                        return
                    n_class_corrected = 0
                    total_classification_loss = 0
                    total_jigen_loss = 0
                    total_sum_loss = 0
                    total_samples = 0
                    self.model.train()
            
            n_class_corrected = 0
            total_classification_loss = 0
            total_jigen_loss = 0
            total_sum_loss = 0
            total_samples = 0
    
    def evaluate(self, n_iter):
        self.model.eval()
        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels, jigen_labels) in enumerate(self.val_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                predicted_jigen_classes, predicted_classes = self.model(samples)
                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        self.writer.add_scalar('Accuracy/validate', 100. * n_class_corrected / len(self.val_loader.dataset), n_iter)
        self.writer.add_scalar('Loss/validate', total_classification_loss / len(self.val_loader.dataset), n_iter)
        logging.info('Val set: Accuracy: {}/{} ({:.0f}%)\tLoss: {:.6f}'.format(n_class_corrected, len(self.val_loader.dataset),
            100. * n_class_corrected / len(self.val_loader.dataset), total_classification_loss / len(self.val_loader.dataset)))
        self.early_stopping(total_classification_loss / len(self.val_loader.dataset), self.model)            
            
    def test(self):
        self.model.eval()
        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels, jigen_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                predicted_jigen_classes, predicted_classes = self.model(samples)
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.0f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))