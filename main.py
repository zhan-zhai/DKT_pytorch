import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Model.DKT import DKT
from Model.DKT_LSTM import DKT_LSTM
from Model.SAKT import SAKT
from Model.NPA import NPA
from Model.DKVMN import DKVMN
import torch.nn as nn
from data_loader import DataReader
from Dataset.DKTDataset import DKTDataset
from Dataset.OtherDataset import OtherDataset
from tqdm import tqdm

from tqdm.contrib import tzip
from sklearn.metrics import roc_auc_score

from itertools import cycle
from transfer.mmd import mmd_rbf_noaccelerate
from transfer.mmd import rbf_mmd

import matplotlib.pyplot as plt
from datetime import datetime
import sys


def train_epoch(model, train_iterator, optim, schedule, criterion,parsers, device):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()
        optim.zero_grad()
        
        
        if parsers.model!='DKT_LSTM':
            output,_ = model(x,target_id)
#             target_mask = (target_id != 0)
#             output = torch.gather(output, 2, target_id.unsqueeze(2) )
#             output = torch.masked_select(output.squeeze(2), target_mask)
#             label = torch.masked_select(label, target_mask)
        else:
            output,_,_ = model(x,target_id,x)
            
#         print(output.shape,label.shape)
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)

    loss = np.average(train_loss)

    return loss, acc, auc

def transfer_train_epoch(model, train_iterator,src_train_iterator, optim,schedule, criterion,parsers, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tzip(train_iterator,cycle(src_train_iterator))
#     src_iter = iter(src_train_iterator)
#     tbar = tqdm(train_iterator)
    for (item,src_item) in tbar:
#         try:
#             src_item = next(src_iter)
#         except StopIteration:
#             src_iter = iter(src_train_iterator)
#             src_item = next(src_iter)
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()
        src_x = src_item[0].to(device).long()
        src_target_id = src_item[1].to(device).long()
        src_label = src_item[2].to(device).float()
        
        optim.zero_grad()
        
        
        if parsers.model!='DKT_LSTM':
            output,output_mmd = model(x,target_id)
            src_output,src_output_mmd = model(src_x,src_target_id)
#             print(output.shape,src_output.shape)
#             mmd_loss = mmd_rbf_noaccelerate(src_output,output)
#             target_mask = (target_id != 0)
#             src_target_mask = (src_target_id != 0)
#             output = torch.gather(output, 2, target_id.unsqueeze(2) )
#             output = torch.masked_select(output.squeeze(2), target_mask)
#             label = torch.masked_select(label, target_mask)
#             src_output = torch.gather(src_output, 2, src_target_id.unsqueeze(2))
#             src_output = torch.masked_select(src_output.unsqueeze(2), src_target_mask)
#             src_label = torch.masked_select(src_label, src_target_mask)
#             print(output.shape,src_output.shape)
        else:
            output,output_mmd,src_output_mmd = model(x,target_id,src_x)
#             src_output = model(src_x,src_target_id)
#             print(output.shape,src_output.shape)
#             mmd_loss = mmd_rbf_noaccelerate(src_output,output)
        
#         print(output_mmd.shape,src_output_mmd.shape)
        mmd_loss = rbf_mmd(output_mmd,src_output_mmd,1)
        loss = criterion(output, label)
        loss += mmd_loss * parsers.lamda
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

#         tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)

    loss = np.average(train_loss)

    return loss, acc, auc


def val_epoch(model, val_iterator, criterion,parsers, device="cpu"):
    model.eval()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(val_iterator)
    for item in tbar:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()

        if parsers.model!='DKT_LSTM':
            with torch.no_grad():
                output,_ = model(x,target_id)
#             target_mask = (target_id != 0)
#             output = torch.gather(output, 2, target_id.unsqueeze(2))
#             output = torch.masked_select(output.squeeze(2), target_mask)
#             label = torch.masked_select(label, target_mask)
        else:
            with torch.no_grad():
                output,_,_ = model(x,target_id,x)
#             output = torch.gather(output, -1, target_id)
            
        loss = criterion(output, label)
        train_loss.append(loss.item())

        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.average(train_loss)

    return loss, acc, auc

def plotAucCurve(epochs,aucs,filename):
    # 保存 AUC 曲线图
    plt.figure()
    plt.plot(range(1, epochs+1), aucs)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC Curve')
    plt.savefig(filename)


# 解析传入的参数
parser = argparse.ArgumentParser(description='myDemo')
parser.add_argument('--batch_size',type=int,default=128,metavar='N',help='number of batch size to train (defauly 64 )')
parser.add_argument('--epochs',type=int,default=30,metavar='N',help='number of epochs to train (defauly 10 )')
parser.add_argument('--lr',type=float,default=0.001,help='number of learning rate')
parser.add_argument('--data_dir', type=str, default='./data/',help="the data directory, default as './data")
parser.add_argument('--hidden_size',type=int,default=100,help='the number of the hidden-size')
parser.add_argument('--max_step',type=int,default=100,help='the number of max step')
parser.add_argument('--num_layers',type=int,default=1,help='the number of layers')
parser.add_argument('--separate_char',type=str,default=',',help='分隔符')
parser.add_argument('--min_step',type=int,default=10,help='the number of min step')

# parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--input_dim', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.2)

# DKVMN
parser.add_argument('--key_dim', type=int, default=50)
parser.add_argument('--value_dim', type=int, default=100)
parser.add_argument('--summary_dim', type=int, default=50)
parser.add_argument('--concept_num', type=int, default=50)

# NPA
parser.add_argument('--attention_dim', type=int, default=100)
parser.add_argument('--fc_dim', type=int, default=200)

# SAKT
parser.add_argument('--num_head', type=int, default=5)

# 源域数据集的题目数量要多于目标域数据集，否则会产生越界错误
parser.add_argument('--lamda', type=float, default=0.5)
parser.add_argument('--mmd', type=bool, default=True)
parser.add_argument('--src_dataset', type=str, default='synthetic', help='which dataset to transfer')
parser.add_argument('--src_train_file', type=str, default='train_set.csv',
                    help="train data file, default as 'train_set.csv'.")
parser.add_argument('--src_test_file', type=str, default='test_set.csv',
                    help="test data file, default as 'test_set.csv'.")
parser.add_argument('--src_n_question', type=int, default=50, help='the number of unique questions in the dataset')
parser.add_argument('--mmd_batch_size',type=int,default=128,metavar='N',help='number of batch size to train tramsfer(defauly 64 )')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    dataset = 'ASSISTments2015'  #  ASSISTments2009 / ASSISTments2015 /  synthetic / statics2011 / junyiacademy / EDNet /KDDCup2010

    model = 'SAKT'  # DKT /
    parser.add_argument('--model', type=str, default='SAKT', help='which model to train')

    if dataset == 'ASSISTments2009':
        parser.add_argument('--dataset', type=str, default='ASSISTments2009', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSISTments2009/',
                            help="train data file, default as './ASSISTments2009/'.")
        parser.add_argument('--n_question', type=int, default=124, help='the number of unique questions in the dataset')

    elif dataset == 'ASSISTments2015':
        parser.add_argument('--dataset', type=str, default='ASSISTments2015', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSISTments2015/',
                            help="train data file, default as './ASSISTments2015/'.")
        parser.add_argument('--n_question', type=int, default=100, help='the number of unique questions in the dataset')
        
    elif dataset == 'ASSISTments2017':
        parser.add_argument('--dataset', type=str, default='ASSISTments2017', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSISTments2017/',
                            help="train data file, default as './ASSISTments2017/'.")
        parser.add_argument('--n_question', type=int, default=102, help='the number of unique questions in the dataset')
    
    elif dataset == 'ASSISTmentsChall':
        parser.add_argument('--dataset', type=str, default='ASSISTmentsChall', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./ASSISTmentsChall/',
                            help="train data file, default as './ASSISTmentsChall/'.")
        parser.add_argument('--n_question', type=int, default=102, help='the number of unique questions in the dataset')

    elif dataset == 'synthetic':
        parser.add_argument('--dataset', type=str, default='synthetic', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./synthetic/',
                            help="train data file, default as './synthetic/'.")
        parser.add_argument('--n_question', type=int, default=50, help='the number of unique questions in the dataset')

    elif dataset == 'statics2011':
        parser.add_argument('--dataset', type=str, default='statics2011', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./statics2011/',
                            help="train data file, default as './statics2011/'.")
        parser.add_argument('--n_question', type=int, default=1224, help='the number of unique questions in the dataset')


    elif dataset =='junyiacademy':
        parser.add_argument('--dataset', type=str, default='junyiacademy', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='problem_train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='problem_test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./junyiacademy/',
                            help="train data file, default as './junyiacademy/'.")
        # parser.add_argument('--n_question', type=int, default=1326, help='the number of unique questions in the dataset')
        parser.add_argument('--n_question', type=int, default=25784, help='the number of unique questions in the dataset')

    elif dataset == 'EDNet':
        parser.add_argument('--dataset', type=str, default='EDNet', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./EDNet/',
                            help="train data file, default as './EDNet/'.")
        parser.add_argument('--n_question', type=int, default=13168, help='the number of unique questions in the dataset')

    elif dataset == 'KDDCup2010':
        parser.add_argument('--dataset', type=str, default='KDDCup2010', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./KDDCup2010/',
                            help="train data file, default as './KDDCup2010/'.")
        parser.add_argument('--n_question', type=int, default=661, help='the number of unique questions in the dataset')
    
    elif dataset == 'transfer':
        parser.add_argument('--dataset', type=str, default='transfer', help='which dataset to train')
        parser.add_argument('--train_file', type=str, default='train_set.csv',
                            help="train data file, default as 'train_set.csv'.")
        parser.add_argument('--test_file', type=str, default='test_set.csv',
                            help="train data file, default as 'test_set.csv'.")
        parser.add_argument('--save_dir_prefix', type=str, default='./transfer/',
                            help="train data file, default as './transfer/'.")
        parser.add_argument('--n_question', type=int, default=678, help='the number of unique questions in the dataset')


#     # 解析参数
    parsers = parser.parse_args()

    print("parser:",parsers)
    # train 路径 和 test 路径
    print(f'loading Dataset  {parsers.dataset}...')
    train_path = parsers.data_dir + parsers.dataset + '/' + parsers.train_file
    test_path = parsers.data_dir + parsers.dataset + '/' + parsers.test_file
    src_train_path = parsers.data_dir + parsers.src_dataset + '/' + parsers.src_train_file
    src_test_path = parsers.data_dir + parsers.src_dataset + '/' + parsers.src_test_file
    train = DataReader(path=train_path, separate_char=parsers.separate_char)
    train_set = train.load_data()
    test = DataReader(path=test_path,separate_char=parsers.separate_char)
    test_dataset = test.load_data()
    src_train = DataReader(path=src_train_path, separate_char=parsers.separate_char)
    src_train_set = src_train.load_data()
    src_test = DataReader(path=src_test_path, separate_char=parsers.separate_char)
    src_test_set = src_test.load_data()

    train_set = pd.DataFrame(train_set,columns=['user_id','skill_id','correct']).set_index('user_id')
    test_dataset = pd.DataFrame(test_dataset, columns=['user_id', 'skill_id', 'correct']).set_index('user_id')
    src_train_set = pd.DataFrame(src_train_set,columns=['user_id','skill_id','correct']).set_index('user_id')
    src_test_set = pd.DataFrame(src_test_set,columns=['user_id','skill_id','correct']).set_index('user_id')

    if parsers.model == 'DKT':
        train_dataset = DKTDataset(group=train_set,n_skill=parsers.n_question,max_seq=parsers.max_step,min_step=parsers.min_step)
        test_dataset = DKTDataset(test_dataset, n_skill=parsers.n_question, max_seq=parsers.max_step, min_step=parsers.min_step)
        src_train_dataset = DKTDataset(group=src_train_set,n_skill=parsers.src_n_question,max_seq=parsers.max_step,min_step=parsers.min_step)
        src_test_dataset = DKTDataset(group=src_test_set,n_skill=parsers.src_n_question,max_seq=parsers.max_step,min_step=parsers.min_step)
    else:
        train_dataset = OtherDataset(group=train_set,n_skill=parsers.n_question,max_seq=parsers.max_step,min_step=parsers.min_step)
        test_dataset = OtherDataset(test_dataset, n_skill=parsers.n_question, max_seq=parsers.max_step, min_step=parsers.min_step)
        src_train_dataset = OtherDataset(group=src_train_set,n_skill=parsers.src_n_question,max_seq=parsers.max_step,min_step=parsers.min_step)
        src_test_dataset = OtherDataset(group=src_test_set,n_skill=parsers.src_n_question,max_seq=parsers.max_step,min_step=parsers.min_step)
    # print(train_dataset.__dict__)
    # 使用固定缓冲区 加快Tensor复制的速度，并且可以使用异步的方式进行复制

    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    full_dataset = train_dataset + test_dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
# #     src_size = int(0.1 * len(src_train_dataset))
    train_dataset,test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
#     left = len(src_train_dataset) - src_size
#     src_train_dataset,_ = torch.utils.data.random_split(src_train_dataset, [src_size,left])
    
    # 如果运行内存不够建议减低num_workers的值
    train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=0,drop_last = True, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=0,drop_last = True, **dataloader_kwargs)
    src_train_dataloader = DataLoader(src_train_dataset, batch_size=parsers.mmd_batch_size, shuffle=True, num_workers=0,drop_last = True, **dataloader_kwargs)
    src_test_dataloader = DataLoader(src_test_dataset, batch_size=parsers.mmd_batch_size, shuffle=True, num_workers=0, **dataloader_kwargs)
    
#     if parsers.mmd:
#         test_dataloader = src_test_dataloader
    # 使用GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")
    print(device)

    if parsers.model == 'DKT':
        model = DKT(input_dim=parsers.n_question*2,hidden_dim=parsers.hidden_size,layer_dim=parsers.num_layers,output_dim = parsers.n_question,mmd = parsers.mmd)
    if parsers.model == 'DKT_LSTM':
        model = DKT_LSTM(parsers.input_dim, parsers.hidden_dim,parsers.num_layers, parsers.n_question, parsers.dropout,mmd = parsers.mmd)

    elif parsers.model == 'DKVMN':
        model = DKVMN(parsers.key_dim, parsers.value_dim, parsers.summary_dim, parsers.n_question,
                      parsers.concept_num)

    elif parsers.model == 'NPA':
        model = NPA(parsers.input_dim, parsers.hidden_dim, parsers.attention_dim, parsers.fc_dim,
                    parsers.num_layers, parsers.n_question, parsers.dropout)

    elif parsers.model == 'SAKT':
        model = SAKT(parsers.hidden_dim, parsers.n_question, parsers.num_layers,
                     parsers.num_head, parsers.dropout, seq_size=parsers.max_step)
    
    print(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=parsers.lr)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
    # nn.BCELoss() 与 nn.BCEWithLogitsLoss() 区别在于 前者进行了sigmoid计算
    criterion = nn.BCEWithLogitsLoss()

    criterion.to(device)

    weight_path = './savedModel/'
    if parsers.mmd:
        weight_path = './savedMMDModel/'

    aucs = []
    best_model = model
    best_auc = 0

    for epoch in range(parsers.epochs):
        if parsers.mmd:
            train_loss, train_acc, train_auc = transfer_train_epoch(model, train_dataloader, src_train_dataloader,optimizer, schedule, criterion,parsers, device)
        else:
            train_loss, train_acc, train_auc = train_epoch(model, train_dataloader, optimizer, schedule, criterion,parsers, device)

        cur_weight = model.state_dict()
        if parsers.mmd:
            torch.save(cur_weight, f'{weight_path}{parsers.model}/{parsers.dataset}_{parsers.src_dataset}_{epoch}.pt')
        else:
            torch.save(cur_weight, f'{weight_path}{parsers.model}/{parsers.dataset}_{epoch}.pt')
        
        print("epoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, train_loss, train_acc, train_auc))

        val_loss, avl_acc, val_auc = val_epoch(model, test_dataloader, criterion,parsers, device)
        print("epoch - {} test_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, val_loss, avl_acc, val_auc))
        
        aucs.append(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model
        
        schedule.step()
    aucs = np.array(aucs)
    if parsers.mmd:
        f = open(f"./savedMMDModel/{parsers.model}/best.txt","a")
    else:
        f = open(f"./savedModel/{parsers.model}/best.txt","a")
    print("best auc - {:.3f} of epoch - {} using {} -- learning_rate - {} -- batch_size - {} -- seq_size - {}".format(np.max(aucs),np.argmax(aucs),parsers.dataset,parsers.lr,parsers.batch_size,parsers.max_step))
    if parsers.mmd:
        print("best auc - {:.3f} of epoch - {} using {} to {} -- learning_rate - {} -- batch_size - {} -- seq_size - {}".format(np.max(aucs),np.argmax(aucs),parsers.dataset,parsers.src_dataset,parsers.lr,parsers.batch_size,parsers.max_step),file=f)
    else:
        print("best auc - {:.3f} of epoch - {} using {} -- learning_rate - {} -- batch_size - {} -- seq_size - {}".format(np.max(aucs),np.argmax(aucs),parsers.dataset,parsers.lr,parsers.batch_size,parsers.max_step),file=f)
        
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'{weight_path}{parsers.model}/{parsers.dataset}_{timestamp}.png'
    if parsers.mmd:
        filename = f'{weight_path}{parsers.model}/{parsers.dataset}_{parsers.src_dataset}_{timestamp}.png'
    plotAucCurve(parsers.epochs,aucs,filename)
    
#     best_model = DKVMN(parsers.key_dim, parsers.value_dim, parsers.summary_dim, parsers.n_question,parsers.concept_num)
#     best_model.load_state_dict(torch.load('./fineTuneModel/DKVMN/ASSISTments2009_ASSISTments2015_46.pt'))
    if not parsers.mmd:
        sys.exit()
        
    for param in best_model.parameters():
        param.requires_grad = False
    best_model.classifier.weight.requires_grad = True
    best_model.classifier.bias.requires_grad = True
    if parsers.model=='DKT_LSTM':
        best_model.classifier = torch.nn.Linear(parsers.hidden_dim, parsers.src_n_question)
    
    fine_tune_aucs = []
    for epoch in range(parsers.epochs):
        train_loss, train_acc, train_auc = train_epoch(best_model, src_train_dataloader, optimizer, schedule, criterion,parsers, device)
        cur_weight = best_model.state_dict()
        torch.save(cur_weight, f'./fineTuneModel/{parsers.model}/{parsers.dataset}_{parsers.src_dataset}_{epoch}.pt')
        print("epoch - {} fine-tune train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, train_loss, train_acc, train_auc))
        val_loss, avl_acc, val_auc = val_epoch(best_model, src_test_dataloader, criterion,parsers, device)
        print("epoch - {} fine-tune test_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, val_loss, avl_acc, val_auc))
        fine_tune_aucs.append(val_auc)
        schedule.step()
    f = open(f"./fineTuneModel/{parsers.model}/best.txt","a")
    print("best auc - {:.3f} of epoch - {} using {} to {} -- learning_rate - {} -- batch_size - {} -- seq_size - {}".format(np.max(fine_tune_aucs),np.argmax(fine_tune_aucs),parsers.dataset,parsers.src_dataset,parsers.lr,parsers.batch_size,parsers.max_step))
    print("best auc - {:.3f} of epoch - {} using {} to {} -- learning_rate - {} -- batch_size - {} -- seq_size - {}".format(np.max(fine_tune_aucs),np.argmax(fine_tune_aucs),parsers.dataset,parsers.src_dataset,parsers.lr,parsers.batch_size,parsers.max_step),file=f)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'./fineTuneModel/{parsers.model}/{parsers.dataset}_{parsers.src_dataset}_{timestamp}.png'
    plotAucCurve(parsers.epochs,fine_tune_aucs,filename)
    
    