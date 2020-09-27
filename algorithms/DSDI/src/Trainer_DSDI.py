import os
import logging
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from algorithms.DSDI.src.dataloaders import dataloader_factory
from algorithms.DSDI.src.models import model_factory
from torch.optim.lr_scheduler import StepLR

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class Class_Classifier(nn.Module):
    def __init__(self, classes):
        super(Class_Classifier, self).__init__()
        self.class_classifier = nn.Linear(256, classes)
        self.dropout= nn.Dropout(p = 0.2)
    
    def forward(self, ds_z, di_z):
        ds_y = self.class_classifier(GradReverse.apply(self.dropout(ds_z)))
        di_y = self.class_classifier(self.dropout(di_z))
        return ds_y, di_y

class Domain_Classifier(nn.Module):
    def __init__(self, domain_classes):
        super(Domain_Classifier, self).__init__()
        self.class_classifier = nn.Linear(256, domain_classes)
        self.dropout= nn.Dropout(p = 0.2)

    def forward(self, ds_z, di_z):
        ds_y = self.class_classifier(self.dropout(ds_z))
        di_y = self.class_classifier(GradReverse.apply(self.dropout(di_z)))
        return ds_y, di_y

class Classifier(nn.Module):
    def __init__(self, classes):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Linear(512, classes)
        self.dropout= nn.Dropout(p = 0.2)
    
    def forward(self, x):
        # y = self.class_classifier(x)
        y = self.class_classifier(self.dropout(x))
        return y 

class Trainer_DSDI:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.writer = self.set_writer(log_dir = "algorithms/" + self.args.algorithm + "/results/tensorboards/" + self.args.exp_name + "/")
        self.train_loader = DataLoader(dataloader_factory.get_train_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_train_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.val_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.src_val_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.test_loader = DataLoader(dataloader_factory.get_test_dataloader(self.args.dataset)(src_path = self.args.src_data_path, meta_filenames = self.args.target_test_meta_filenames), batch_size = self.args.batch_size, shuffle = True)
        self.model = model_factory.get_model(self.args.model)(classes = self.args.n_classes).to(self.device)
        
        self.classifer = Classifier(7).to(self.device)
        
        self.class_classifier = Class_Classifier(7).to(self.device)
        self.domain_classifer = Domain_Classifier(3).to(self.device)

        # optimizer_dict = [
		# 	{"params": filter(lambda p: p.requires_grad, self.model.parameters())},
		# 	{"params": filter(lambda p: p.requires_grad, self.classifer.parameters())},
		# 	{"params": filter(lambda p: p.requires_grad, self.class_classifier.parameters())},
		# 	{"params": filter(lambda p: p.requires_grad, self.domain_classifer.parameters())},
		# ]
        optimizer_dict = list(self.model.parameters()) + list(self.classifer.parameters()) + list(self.class_classifier.parameters()) + list(self.domain_classifer.parameters())

        self.optimizer = torch.optim.SGD(optimizer_dict, lr = self.args.learning_rate, weight_decay=.0005, momentum=.9, nesterov=False)
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
        self.classifer.train()
        self.class_classifier.train()
        self.domain_classifer.train()
        
        n_class_corrected = 0
        total_classification_loss = 0
        total_samples = 0
        for epoch in range(self.args.epochs):
            for iteration, (samples, labels, domain_labels) in enumerate(self.train_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                self.optimizer.zero_grad()
                
                # predicted_classes = self.model(samples)
                ds_z, di_z, x = self.model(samples)
                predicted_classes = self.classifer(x)

                classification_loss = self.criterion(predicted_classes, labels)
                total_classification_loss += classification_loss.item()

                predicted_classes_ds, predicted_classes_di = self.class_classifier(ds_z, di_z)
                predicted_domain_ds, predicted_domain_di = self.domain_classifer(ds_z, di_z)

                predicted_classes_ds_loss = self.criterion(predicted_classes_ds, labels)
                predicted_classes_di_loss = self.criterion(predicted_classes_di, labels)
                predicted_domain_ds_loss = self.criterion(predicted_domain_ds, domain_labels)
                predicted_domain_di_loss = self.criterion(predicted_domain_di, domain_labels)
                
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
                total_samples += len(samples)

                total_loss = classification_loss + predicted_classes_ds_loss + predicted_classes_di_loss + predicted_domain_ds_loss + predicted_domain_di_loss
                total_loss.backward()
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
                    self.classifer.train()
                    self.class_classifier.train()
                    self.domain_classifer.train()
            
            n_class_corrected = 0
            total_classification_loss = 0
            total_samples = 0
    
    def evaluate(self, n_iter):
        self.model.eval()
        self.classifer.eval()
        self.class_classifier.eval()
        self.domain_classifer.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.val_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                # predicted_classes = self.model(samples)
                ds_z, di_z, x = self.model(samples)
                predicted_classes = self.classifer(x)
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
            torch.save(self.classifer.state_dict(), self.checkpoint_name + '_classifer.pt')           
            
    def test(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name + '.pt'))
        self.classifer.load_state_dict(torch.load(self.checkpoint_name + '_classifer.pt'))
        
        self.model.eval()
        self.classifer.eval()
        self.class_classifier.eval()
        self.domain_classifer.eval()

        n_class_corrected = 0
        with torch.no_grad():
            for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
                samples, labels, domain_labels = samples.to(self.device), labels.to(self.device), domain_labels.to(self.device)
                # predicted_classes = self.model(samples)
                ds_z, di_z, x = self.model(samples)
                predicted_classes = self.classifer(x)
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info('Test set: Accuracy: {}/{} ({:.0f}%)'.format(n_class_corrected, len(self.test_loader.dataset), 
            100. * n_class_corrected / len(self.test_loader.dataset)))