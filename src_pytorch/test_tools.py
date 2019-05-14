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



def model_predict(mod, dataloaders):
    since = time.time()
    acc = 0
    recall = 0
    batches = len(dataloaders[TEST])

    mod.train(False)
    mod.eval()

    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\rBatch {}/{}".format(i, test_batches))

        inputs, labels = data
        inputs, labels = Variable(inputs.to(device, dtype=torch.float), volatile=True), Variable(labels.to(device, dtype=torch.long), volatile=True)
        outputs = mod(inputs)

        _, preds = torch.max(outputs.data, 1)
        acc += torch.sum(preds == labels.data)
        recall += recall_score(labels.data, preds)
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    acc = float(acc.cpu()[0]) / dataset_sizes[TEST]
    recall = float(recall / dataset_sizes[TEST])
    # elapsed_time = time.time() - since

    return recall, acc

