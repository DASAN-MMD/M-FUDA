# https://github.com/easezyc/deep-transfer-learning/tree/master/MUDA
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import ResNet50 as models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import torch.utils.model_zoo as model_zoo
import time
import pickle
import argparse
from sklearn.manifold import TSNE
from torchvision.ops import nms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 10 #30
iteration = 300
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
seed = 8
log_interval = 1
l2_decay = 5e-4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('bending', 'falling', 'lie_down', 'running', 'sit_down', 'stand_up', 'walking')

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)


train_src1_dataset_path= "./Dataset_S1_S2_S3/training1/training"
train_src2_dataset_path="./Dataset_S1_S2_S3/training2/training"
train_src3_dataset_path="./Dataset_S1_S2/training/training"
test_dataset_path="./Dataset_S1_S2_S3/training3/training"

mean=[0.7290,0.8188,0.6578]
std=[0.2965,0.1467,0.2864]


train_transforms=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

train_src1_dataset=torchvision.datasets.ImageFolder(root=train_src1_dataset_path,transform=train_transforms)
train_src2_dataset=torchvision.datasets.ImageFolder(root=train_src2_dataset_path,transform=train_transforms)
train_src3_dataset=torchvision.datasets.ImageFolder(root=train_src3_dataset_path,transform=train_transforms)
test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=train_transforms)

np.random.seed(rand_seed)
idxs=np.random.permutation(len(test_dataset))

source1_loader =torch.utils.data.DataLoader(train_src1_dataset,batch_size=batch_size , num_workers=4, shuffle=True)
source2_loader =torch.utils.data.DataLoader(train_src2_dataset,batch_size=batch_size , num_workers=4, shuffle=True)
source3_loader =torch.utils.data.DataLoader(train_src3_dataset,batch_size=batch_size , num_workers=4, shuffle=True)


test_sampler1=SubsetRandomSampler(idxs[:81])
target_train_loader = DataLoader(test_dataset, batch_size=batch_size , num_workers=4, sampler=test_sampler1) #Use for model training with 0.8 per target samples

test_sampler2=SubsetRandomSampler(idxs[81:111])
target_validation_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, sampler=test_sampler2) # Use for model training with 0.6 per target samples

test_sampler3=SubsetRandomSampler(idxs[111:])
target_test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4 , sampler=test_sampler3) # Use for model testing with 0.2 per target samples

def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    target_iter = iter(target_train_loader)
    correct = 0
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc_son1.parameters(), 'lr': lr[1]},
        {'params': model.cls_fc_son2.parameters(), 'lr': lr[1]},
        {'params': model.cls_fc_son3.parameters(), 'lr': lr[1]},
        {'params': model.sonnet1.parameters(), 'lr': lr[1]},
        {'params': model.sonnet2.parameters(), 'lr': lr[1]},
        {'params': model.sonnet3.parameters(), 'lr': lr[1]},
    ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration + 1):
        model.train()
        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[5]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[6]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)

        try:
            #source_data, source_label = source1_iter.next()
            source_data, source_label = next(source1_iter)
        except Exception as err:
            source1_iter = iter(source1_loader)
            #source_data, source_label = source1_iter.next()
            source_data, source_label = next(source1_iter)
        try:
            #target_data, __ = target_iter.next()
            target_data, __ = next(target_iter)
        except Exception as err:
            target_iter = iter(target_train_loader)
            #target_data, __ = target_iter.next()
            target_data, __ = next(target_iter)
        if cuda:
            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)
        #source_data, source_label = Variable(source_data), Variable(source_label)
        #target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss, csa_loss = model(source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss) + (0.01 * csa_loss)
        #loss = cls_loss + gamma * (l1_loss) + (0.01 * csa_loss)
        loss.backward()
        optimizer.step()

        try:
            #source_data, source_label = source2_iter.next()
            source_data, source_label = next(source2_iter)
        except Exception as err:
            source2_iter = iter(source2_loader)
            #source_data, source_label = source2_iter.next()
            source_data, source_label = next(source2_iter)
        try:
            #target_data, __ = target_iter.next()
            target_data, __ = next(target_iter)
        except Exception as err:
            target_iter = iter(target_train_loader)
            #target_data, __ = target_iter.next()
            target_data, __ = next(target_iter)
        if cuda:
            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss, csa_loss = model(source_data, target_data, source_label, mark=2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss) + (0.01 * csa_loss)
        #loss = cls_loss + gamma * (l1_loss) + (0.01 * csa_loss)
        loss.backward()
        optimizer.step()

        try:
            #source_data, source_label = source3_iter.next()
            source_data, source_label = next(source3_iter)
        except Exception as err:
            source3_iter = iter(source3_loader)
            #source_data, source_label = source3_iter.next()
            source_data, source_label = next(source3_iter)
        try:
            #target_data, __ = target_iter.next()
            target_data, __ = next(target_iter)
        except Exception as err:
            target_iter = iter(target_train_loader)
            #target_data, __ = target_iter.next()
            target_data, __ = next(target_iter)
        if cuda:
            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss, csa_loss = model(source_data, target_data, source_label, mark=3)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss) + (0.01 * csa_loss)
        #loss = cls_loss + gamma * (l1_loss) + (0.01 * csa_loss)
        loss.backward()
        optimizer.step()

        t_correct, test_loss, acc = evaluate(model)
        train_losses.append(test_loss)
        train_accuracy.append(acc * 100)
        if t_correct > correct:
            correct = t_correct
        print('Train source iter: {} [({:.0f}%)]\tTraining Loss : {:.4f}'.format(i, 100. * i / iteration, test_loss))
        print(source1_name, source2_name, source3_name, "to", target_name, "%s max correct:" % target_name,
              correct.item(), "\n")


def evaluate(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct_src1 = 0
    correct_src2 = 0
    correct_src3 = 0
    with torch.no_grad():
        for data, target in test_loader2:
            if cuda:
                data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3 = model(data)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)

            pred = (pred1 + pred2 + pred3) / 3
            #pred = pred1
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()  # sum up batch loss
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]  # get the index of the max log-probability
            correct_src1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]  # get the index of the max log-probability
            correct_src2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred3.data.max(1)[1]  # get the index of the max log-probability
            correct_src3 += pred.eq(target.data.view_as(pred)).cpu().sum()

        acc = correct / (len(test_loader2)*batch_size )
        test_loss /= (len(test_loader2)*batch_size )
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, (len(test_loader2)*batch_size ),
            100. * correct / (len(test_loader2)*batch_size )))
        print('\nsource1 accnum {}, source2 accnum {}，source3 accnum {}'.format(correct1, correct2, correct3))
    return correct, test_loss, acc

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct_src1 = 0
    correct_src2 = 0
    correct_src3 = 0
    #size = 0
    all_features = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            all_labels = torch.cat((all_labels, target), dim=0)  # For Plotting Purpose in CMT & Hist
            pred1, pred2, pred3 = model(data)

            pred_feat = (pred1 + pred2 + pred3) / 3
            #all_features.append(pred_feat)
            all_features = torch.cat((all_features, pred_feat), dim=0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)

            pred = (pred1 + pred2 + pred3) / 3
            #pred = pred1
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()  # sum up batch loss
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            all_preds = torch.cat((all_preds, pred), dim=0)  # For Plotting Purpose in CMT
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]  # get the index of the max log-probability
            correct_src1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]  # get the index of the max log-probability
            correct_src2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred3.data.max(1)[1]  # get the index of the max log-probability
            correct_src3 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= (len(target_test_loader) * batch_size )
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, (len(target_test_loader) * batch_size ),
            100. * correct / (len(target_test_loader) * batch_size )))
        print('\nsource1 accnum {}, source2 accnum {}，source3 accnum {}'.format(correct1, correct2, correct3))
    return all_labels, all_preds, all_features


if __name__ == '__main__':
    model = models.MFUDA(num_classes=7)
    #model = models.DDCNet(num_classes=7)
    #print(model)
    if cuda:
        model.to(device)

    training_start_time = time.time()
    num_epoch = train(model)
    print(num_epoch)
    time1 = time.time() - training_start_time
    print('Training_Time:{:.2f}'.format(time1))

    all_labels, all_preds, all_features = test(model)

    print(classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy(), target_names=classes, zero_division=0))
    print('f1_score(micro):{:.2f}'.format(f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='micro')))
    print('f1_score(macro):{:.2f}'.format(f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')))
