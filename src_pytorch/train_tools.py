import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
import torch.utils.data as utils
from tqdm import tqdm
from os.path import isfile, join
from os import listdir
import pandas as pd
import time, copy


def train_model_train(model, criterion, optimizer, dataloaders):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss, running_corrects


def train_model_evaluate(model, criterion, dataloaders):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    out = []
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            out.append(outputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    return running_loss, running_corrects


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    logs = pd.DataFrame(columns=['val_acc', 'val_loss', 'train_loss', 'train_acc'])
    val_acc_history, val_loss_history = [], []
    train_acc_history, train_loss_history = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_loss = 0.0, 100.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                running_loss, running_corrects = train_model_train(model, criterion, optimizer, dataloaders)
            else:
                running_loss, running_corrects = train_model_evaluate(model, criterion, dataloaders)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc), val_loss_history.append(epoch_loss)

            if phase == 'train':
                train_acc_history.append(epoch_acc), train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    logs['val_acc'],  logs['val_loss'] = val_acc_history,  val_loss_history
    logs['train_acc'], logs['train_loss'] = train_acc_history, train_loss_history

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, logs