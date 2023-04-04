# -*- coding: utf-8 -*-


import time
import os
import json
import torch as th
th.autograd.set_detect_anomaly(True)
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
th.set_num_threads(2)
import datetime
import numpy as np
import copy
from objectives.nce import C_NCE,NegNCE,NegNCE_f,AdverNCE,NegNCETime,AdverNCETime,RandomNCE,RandomNCETime
from objectives.mll import MaximumLogLikelihood
from objectives.ce import CE,RandomCE,RandomCEAndTime
from objectives.ce import CEAndTime , AdverCE, AdverCETime
from model.sasrec import SASRec
from model.gcsan import GCSAN
from model.srgnn import SRGNN
# from model.c_sasrec import CSASRec
from model.t_sasrec import TimePredSASRec
from model.gru4rec import GRU4Rec
from model.stamp import STAMP
from model.narm import NARM
from model.t_gru4rec import  TimePredGRU4Rec
# ignore weight decay for bias and batch norm
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'batch_norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(inputs, device):
    inputs_gpu = [x.to(device) for x in inputs]
    return inputs_gpu


def prepare_batch2(inputs, device,epoch,round,K):
    epoch = epoch % round

    inputs_gpu = [x.to(device) for x in inputs]
    inputs_gpu[-1] = inputs_gpu[-1][:,epoch*K:(epoch+1)*K]


    return inputs_gpu

def load_q_model(model_save_path,config_path):
    # model_save_path = path+'.pkl'
    # config_path = path +'_config.json'
    with open(config_path,'r') as f: # 噪音模型的config文件
        config = json.load(f)
    loaded_model_dict = th.load(model_save_path)

    if config['model'] == 'SASRec':
        best_model = SASRec(config, config['num_items'])
    elif config['model'] == 'TimePredSASRec':
        best_model = TimePredSASRec(config,config['num_items'])
    elif config['model'] == 'GRU4Rec':
        best_model = GRU4Rec(config,config['num_items'])
    elif config['model'] == 'TimePredGRU4Rec':
        best_model = TimePredGRU4Rec(config,config['num_items'])


    device = config['device']
    # device = 'cpu'
    best_model = best_model.to(device)
    best_model.load_state_dict(loaded_model_dict)
    for p in best_model.parameters(): # 固定噪音模型的参数，不改变
        p.requires_grad = False

    return best_model

def load_new_model(config ):
    if config['model'] == 'SASRec':
        model_obj = SASRec(config, config['num_items'])
    elif config['model'] == 'TimePredSASRec':
        model_obj = TimePredSASRec(config, config['num_items'])
    elif config['model'] == 'GRU4Rec':
        model_obj = GRU4Rec(config,config['num_items'])
    elif config['model'] == 'TimePredGRU4Rec':
        model_obj = TimePredGRU4Rec(config,config['num_items'])
    elif config['model'] == 'NARM':
        model_obj = NARM(config,config['num_items'])
    elif config['model'] == 'STAMP':
        model_obj = STAMP(config,config['num_items'])
    elif config['model'] == 'GCSAN':
        model_obj = GCSAN(config,config['num_items'])
    elif config['model'] == 'SRGNN':
        model_obj = SRGNN(config, config['num_items'])

    model_obj = model_obj.to(config['device'])
    return model_obj







class TrainRunnerTime:
    def __init__(
            self,
            config,
            train_loader,
            test_loader,
            dev_loader,  # TODO
            logger,
            # lr=1e-3,
            # weight_decay=0,
            # patience=3,
    ):

        device = config['device']
        self.config = config

        # if config['fine_tune']:
        #     p_model = load_model2(config['noise_model_path'])
        # else:

        PATH_to_log_dir = config['log_name']

        root_tag = self.config['hyper_combi'] + '/'
        os.makedirs(PATH_to_log_dir + root_tag)
        self.writer = SummaryWriter(PATH_to_log_dir + root_tag)

        self.train_batch_loss_tag = root_tag + 'train/batch_loss/'
        self.train_epoch_loss_tag = root_tag + 'train/epoch_loss/'

        self.train_batch_time_loss_tag = root_tag + 'train/batch_time_loss/'

        self.dev_hit5_tag = root_tag + 'dev/hit5/'
        self.dev_ndcg5_tag = root_tag + 'dev/ndcg5/'
        self.dev_hit10_tag = root_tag + 'dev/hit10/'
        self.dev_ndcg10_tag = root_tag + 'dev/ndcg10/'
        self.dev_hit20_tag = root_tag + 'dev/hit20/'
        self.dev_ndcg20_tag = root_tag + 'dev/ndcg20/'
        self.dev_mae_tag = root_tag + 'dev/mae/'

        self.test_hit5_tag = root_tag + 'test/hit5/'
        self.test_ndcg5_tag = root_tag + 'test/ndcg5/'
        self.test_hit10_tag = root_tag + 'test/hit10/'
        self.test_ndcg10_tag = root_tag + 'test/ndcg10/'
        self.test_hit20_tag = root_tag + 'test/hit20/'
        self.test_ndcg20_tag = root_tag + 'test/ndcg20/'
        self.test_mae_tag = root_tag + 'test/mae/'

        p_model = load_new_model(config)

        self.model = p_model
        self.best_model = copy.deepcopy(p_model.state_dict())

        self.optimizer = optim.Adam(p_model.parameters(), lr=config['learning_rate'])

        self.train_method = config['train_method']


        if self.train_method == 'ce':
            self.criterion = CE()
        elif self.train_method == 'ce_time':
            self.criterion = CEAndTime()
        elif self.train_method == 'random_ce':
            self.criterion = RandomCE(config['K'])
        elif self.train_method == 'random_ce_time':
            self.criterion = RandomCEAndTime(config['K'])
        elif self.train_method == 'random_nce' :
            self.criterion = RandomNCE(config['K'])
        elif self.train_method == 'random_nce_time' :
            self.criterion = RandomNCETime(config['K'])

        elif self.train_method == 'neg_ce':
            self.criterion = AdverCE(config['K'])
            self.q_model = load_q_model(config['noise_model_path'], config['noise_config_path'])
        elif self.train_method == 'neg_ce_time':
            self.criterion = AdverCETime(config['K'])
            self.q_model = load_q_model(config['noise_model_path'], config['noise_config_path'])
        elif self.train_method == 'neg_nce':
            self.criterion = AdverNCE(config['K'])
            self.q_model = load_q_model(config['noise_model_path'], config['noise_config_path'])
        elif self.train_method == 'neg_nce_time':
            self.criterion = AdverNCETime(config['K'])
            self.q_model = load_q_model(config['noise_model_path'], config['noise_config_path'])

        elif self.train_method == 'adver_ce':
            self.criterion = AdverCE(config['K'])
            self.q_model = load_new_model(config)
            for p in self.q_model.parameters():  # 固定噪音模型的参数，不改变
                p.requires_grad = False
        elif self.train_method == 'adver_ce_time':
            self.criterion = AdverCETime(config['K'])
            self.q_model = load_new_model(config)
            for p in self.q_model.parameters():  # 固定噪音模型的参数，不改变
                p.requires_grad = False
        elif self.train_method == 'adver_nce' :
            self.criterion = AdverNCE(config['K'])
            self.q_model = load_new_model(config)
            for p in self.q_model.parameters():  # 固定噪音模型的参数，不改变
                p.requires_grad = False
        elif self.train_method == 'adver_nce_time' :
            self.criterion = AdverNCETime(config['K'])
            self.q_model = load_new_model(config)
            for p in self.q_model.parameters():  # 固定噪音模型的参数，不改变
                p.requires_grad = False

        elif self.train_method == 'mll':

            self.criterion = MaximumLogLikelihood()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dev_loader = dev_loader
        self.device = device
        self.epoch = 0
        self.batch = 0
        self.sec_time = 0

        self.patience = 3
        self.logger = logger
        self.best_ndcg_10 = 0.
        self.best_hr_10 = 0.
        self.early_stop = 0
        # self.last_loss = 0.
        # self.best_loss = 0.
        self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
        self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
        self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20, self.best_test_mae= 0, 0, 0, 0, 0, 0, 0, 0, 0,0





    def evaluate (self, model, data_loader, device,tag):
        model.eval()

        mrr_5 = th.tensor(0.0).to(device)
        hit_5 = th.tensor(0.0).to(device)
        ndcg_5 = th.tensor(0.0).to(device)
        mrr_10 = th.tensor(0.0).to(device)
        hit_10 = th.tensor(0.0).to(device)
        ndcg_10 = th.tensor(0.0).to(device)
        mrr_20 = th.tensor(0.0).to(device)
        hit_20 = th.tensor(0.0).to(device)
        ndcg_20 = th.tensor(0.0).to(device)
        mae = th.tensor(0.0).to(device)
        num_samples = 0
        log2 = th.log(th.tensor(2.)).to(device)
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len, time_seq, time_interval_seq, target_time = prepare_batch(
                    batch, device)

                last_time = th.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
                target_interval = target_time.reshape(-1, 1) - last_time

                logits,predict_intervals = model.calculate_logits_time(item_seq, item_seq_len, time_seq, time_interval_seq, target_time)

                # TODO modified
                # granularity = 24 * 5
                # target_interval = target_interval // granularity
                # target_interval[target_interval > 12] = 12
                # target_interval = target_interval.squeeze(-1)
                # predict_intervals = predict_intervals.argmax(dim = -1)



                mae += th.sum(th.abs(target_interval-predict_intervals))

                batch_size = logits.size(0)
                num_samples += batch_size
                labels = target_id.unsqueeze(-1)

                _, topk = logits.topk(k=5)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_5 += hit_ranks.numel()
                mrr_5 += r_ranks.sum()
                ndcg_5 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=10)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_10 += hit_ranks.numel()
                mrr_10 += r_ranks.sum()
                ndcg_10 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

                _, topk = logits.topk(k=20)
                hit_ranks = th.where(topk == labels)[1] + 1  # 应该是个元组
                r_ranks = 1 / hit_ranks.to(th.float32)
                hit_20 += hit_ranks.numel()
                mrr_20 += r_ranks.sum()
                ndcg_20 += (log2 / th.log(hit_ranks.to(th.float32) + 1)).sum()

        hit_5 = hit_5 / num_samples
        ndcg_5 = ndcg_5 / num_samples
        mrr_5 = mrr_5 / num_samples
        hit_10 = hit_10 / num_samples
        ndcg_10 = ndcg_10 / num_samples
        mrr_10 = mrr_10 / num_samples
        hit_20 = hit_20 / num_samples
        ndcg_20 = ndcg_20 / num_samples
        mrr_20 = mrr_20 / num_samples

        mae = mae/num_samples
        if tag == 'dev':
            self.writer.add_scalar(self.dev_hit5_tag, hit_5, self.batch)
            self.writer.add_scalar(self.dev_hit10_tag, hit_10, self.batch)
            self.writer.add_scalar(self.dev_hit20_tag, hit_20, self.batch)
            self.writer.add_scalar(self.dev_ndcg5_tag, ndcg_5, self.batch)
            self.writer.add_scalar(self.dev_ndcg10_tag, ndcg_10, self.batch)
            self.writer.add_scalar(self.dev_ndcg20_tag, ndcg_20, self.batch)
            self.writer.add_scalar(self.dev_mae_tag, mae, self.batch)
        else:
            self.writer.add_scalar(self.test_hit5_tag, hit_5, self.batch)
            self.writer.add_scalar(self.test_hit10_tag, hit_10, self.batch)
            self.writer.add_scalar(self.test_hit20_tag, hit_20, self.batch)
            self.writer.add_scalar(self.test_ndcg5_tag, ndcg_5, self.batch)
            self.writer.add_scalar(self.test_ndcg10_tag, ndcg_10, self.batch)
            self.writer.add_scalar(self.test_ndcg20_tag, ndcg_20, self.batch)
            self.writer.add_scalar(self.test_mae_tag, mae, self.batch)

        return hit_5, ndcg_5, mrr_5, \
               hit_10, ndcg_10, mrr_10, \
               hit_20, ndcg_20, mrr_20,mae



    def train(self, epochs):

        # mean_loss = 0
        last_loss = float('inf')
        for epoch in range(epochs):
            # starttime = datetime.datetime.now()

            self.model.train()
            total_loss = 0
            count = 0
            for batch in self.train_loader:

                # starttime = datetime.datetime.now()



                self.optimizer.zero_grad()

                

                if self.config['train_method'] == 'ce' or self.config['train_method'] == 'ce_time' or\
                    self.config['train_method'] == 'random_ce' or self.config['train_method'] == 'random_ce_time' or\
                    self.config['train_method'] == 'random_nce' or self.config['train_method'] == 'random_nce_time' or \
                    self.config['train_method'] == 'mll':
                    user_id, item_seq, target_id, item_seq_len, time_seq, time_interval_seq, target_time = prepare_batch(
                        batch, self.device)
                    loss = self.criterion(self.model, item_seq, item_seq_len, target_id, time_seq, time_interval_seq,
                                          target_time)

                if self.config['train_method'] == 'neg_nce' or self.config['train_method'] == 'neg_nce_time' or \
                        self.config['train_method'] == 'neg_ce' or self.config['train_method'] == 'neg_ce_time'     :
                    user_id, item_seq, target_id, item_seq_len, time_seq, time_interval_seq, target_time= prepare_batch( batch, self.device)
                    loss = self.criterion(self.model, self.q_model, item_seq, item_seq_len, target_id, time_seq,
                                          time_interval_seq, target_time)
                    # print('i am in training method...........')
                if self.config['train_method'] == 'adver_nce' or self.config['train_method'] == 'adver_nce_time' or \
                        self.config['train_method'] == 'adver_ce' or self.config['train_method'] == 'adver_ce_time'     :
                    user_id, item_seq, target_id, item_seq_len, time_seq, time_interval_seq, target_time = prepare_batch(
                        batch, self.device)
                    loss = self.criterion(self.model, self.q_model, item_seq, item_seq_len, target_id, time_seq,
                                          time_interval_seq, target_time)
                    state_dict = self.model.state_dict()
                    self.q_model.load_state_dict(state_dict)
                    for p in self.q_model.parameters():  # 固定噪音模型的参数，不改变
                        p.requires_grad = False
                # if th.isnan(loss):
                #     print('i am here')
                self.writer.add_scalar(self.train_batch_loss_tag, loss.item(), self.batch)

                with th.autograd.detect_anomaly():
                    loss.backward()
                # loss.backward()
                # 梯度裁剪

                self.optimizer.step()

                self.batch += 1
                count += len(batch)
                total_loss += loss.item() * len(batch)


            avg_loss = total_loss / count
            self.writer.add_scalar(self.train_epoch_loss_tag, avg_loss, epoch)
            # TODO Early Stop
            # TODO @k

            dev_hit_5, dev_ndcg_5, dev_mrr_5, \
            dev_hit_10, dev_ndcg_10, dev_mrr_10, \
            dev_hit_20, dev_ndcg_20, dev_mrr_20,\
            dev_mae = self.evaluate (self.model, self.dev_loader, self.device,'dev')

            test_hit_5, test_ndcg_5, test_mrr_5, \
            test_hit_10, test_ndcg_10, test_mrr_10, \
            test_hit_20, test_ndcg_20, test_mrr_20,\
            test_mae = self.evaluate (self.model, self.test_loader, self.device,'test')


            if dev_hit_10 > self.best_hr_10 and dev_ndcg_10 > self.best_ndcg_10:
                self.best_hr_10 = dev_hit_10
                self.best_ndcg_10 = dev_ndcg_10
                self.best_model = copy.deepcopy(self.model.state_dict())

                self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20 = \
                    test_hit_5, test_ndcg_5, test_mrr_5, \
                    test_hit_10, test_ndcg_10, test_mrr_10, \
                    test_hit_20, test_ndcg_20, test_mrr_20
                self.best_test_mae = test_mae

                self.early_stop = 0
            else:
                self.early_stop += 1

            self.logger.info('training printing epoch: %d' % epoch)
            self.logger.info('[dev] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f, '
                             'mae: %.5f'% (dev_hit_5, dev_ndcg_5, dev_mrr_5,
                                                                            dev_hit_10, dev_ndcg_10, dev_mrr_10,
                                                                            dev_hit_20, dev_ndcg_20, dev_mrr_20,dev_mae))
            self.logger.info('[test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f, '
                             'mae: %.5f'% (test_hit_5, test_ndcg_5, test_mrr_5,
                                test_hit_10, test_ndcg_10, test_mrr_10,
                                test_hit_20, test_ndcg_20, test_mrr_20,test_mae))

            self.logger.info('[best test] hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                             'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                             'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f, '
                             'mae: %.5f'% (self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5,
                                self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10,
                                self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20,
                                self.best_test_mae))

            if self.early_stop >= self.patience:
                break
        # self.w.close()
        return self.best_test_hit_5, self.best_test_ndcg_5, self.best_test_mrr_5, \
               self.best_test_hit_10, self.best_test_ndcg_10, self.best_test_mrr_10, \
               self.best_test_hit_20, self.best_test_ndcg_20, self.best_test_mrr_20, \
               self.best_test_mae, self.best_hr_10, self.best_ndcg_10
        # TODO 可以返回最好的valid score

    def get_top1(self, model, ):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        top1_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len)

                _, topk = logits.topk(k=1)
                top1_lst.extend(list(topk.cpu().numpy().reshape(-1)))
        return top1_lst

    def get_best_model(self):
        return self.best_model

    def get_topk(self, model, k):
        model.eval()
        device = self.device
        data_loader = self.test_loader

        topk_lst = []
        with th.no_grad():
            for batch in data_loader:
                user_id, item_seq, target_id, item_seq_len = prepare_batch(batch, device)
                logits = model.calculate_logits(item_seq, item_seq_len)

                _, topk = logits.topk(k=k)
                topk_lst.extend(topk.cpu().numpy().tolist())
        return topk_lst
