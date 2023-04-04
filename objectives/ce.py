# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

class CE(nn.Module):

    def __init__(self):
        super(CE, self).__init__()
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, p_model, item_seq, item_seq_len, target_id, time_seq,time_interval_seq,target_time):
        logits = p_model.calculate_logits(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)

        # logits[torch.isnan(logits)] = -1000#TODO 这里把nan变成极小的值
        # logits[logits<=0] = -1000
        loss = self.criterion(logits, target_id)
        return loss


class CEAndTime(nn.Module):

    def __init__(self):
        super(CEAndTime, self).__init__()
        self.type_criterion = nn.CrossEntropyLoss()
        self.time_criterion = nn.MSELoss()
        # self.time_criterion = nn.CrossEntropyLoss()


    def forward(self, p_model, item_seq, item_seq_len, target_id, time_seq,time_interval_seq,target_time):
        logits ,predict_intervals= p_model.calculate_logits_time(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)


        type_loss = self.type_criterion(logits, target_id)


        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        target_interval = target_time.reshape(-1, 1) - last_time

        granularity = 24 * 30*6  # 对于亚马逊这个数据集以半年为单位
        time_loss = self.time_criterion(predict_intervals/granularity, target_interval/granularity)/5  # 用天做单位

        # granularity = 24 * 14  # 对于tmall这个数据集以周为单位
        # time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity)    # 用天做单位
        # target_interval = target_interval//granularity
        # target_interval[target_interval>12] = 12
        # target_interval = target_interval.squeeze(-1).long()
        # time_loss =  self.time_criterion(predict_intervals,target_interval)


        loss = type_loss + time_loss

        return loss


class RandomCE(nn.Module):

    def __init__(self,K):
        super(RandomCE,self).__init__()
        self.K = K
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,p_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time,seq_emb=None,neg_items=None):
        batch_size = item_seq.shape[0]
        """
        Random Sampling
        """

        ones = torch.ones(batch_size,p_model.n_items).to(item_seq.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        # one_hot_noise_logits = noise_logits * one_hot
        one_hot_noise_prob = F.softmax(one_hot, dim=-1) * one_hot
        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)


        pos_neg_items = torch.cat((target_id.reshape(-1,1),neg_items),1)# B,1+K
        pos_neg_scores = p_model.calculate_logits_given_items(item_seq,item_seq_len,pos_neg_items, time_seq ,time_interval_seq ,target_time )
        targets = torch.zeros(batch_size,dtype=torch.long).to(item_seq.device)



        loss = self.criterion(pos_neg_scores,targets)
        return loss


class RandomCEAndTime(nn.Module):

    def __init__(self, K):
        super(RandomCEAndTime, self).__init__()
        self.K = K
        self.type_criterion = nn.CrossEntropyLoss()
        self.time_criterion = nn.MSELoss()

    def forward(self, p_model, item_seq, item_seq_len, target_id, time_seq, time_interval_seq, target_time, seq_emb=None,
                neg_items=None):
        batch_size = item_seq.shape[0]

        """
        samping
        """

        ones = torch.ones(batch_size,p_model.n_items).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        # one_hot_noise_logits = noise_logits * one_hot
        one_hot_noise_prob = F.softmax(one_hot, dim=-1) * one_hot
        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)

        """
        scoring
        """

        pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)  # B,1+K
        pos_neg_scores,predict_intervals = p_model.calculate_logits_time_given_items(item_seq, item_seq_len, pos_neg_items, time_seq,
                                                              time_interval_seq, target_time)
        targets = torch.zeros(batch_size, dtype=torch.long).to(p_model.device)

        type_loss = self.type_criterion(pos_neg_scores, targets)

        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        target_interval = target_time.reshape(-1, 1) - last_time
        granularity = 24 * 30*6  # 对于亚马逊这个数据集以半年为单位
        time_loss = self.time_criterion(predict_intervals/granularity, target_interval/granularity)/5  # 用天做单位

        # granularity = 24 * 14  # 对于tmall这个数据集以周为单位
        # time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity)  # 用天做单位


        loss = type_loss + time_loss
        return loss


class AdverCE(nn.Module):

    def __init__(self,K):
        super(AdverCE,self).__init__()
        self.K = K

        self.criterion = nn.CrossEntropyLoss()


    def forward(self,p_model,q_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time ):

        batch_size = item_seq.shape[0]

        noise_logits  = q_model.calculate_logits (item_seq, item_seq_len, time_seq,time_interval_seq,target_time)
        ones = torch.ones(noise_logits.shape).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        one_hot_noise_prob = F.softmax(noise_logits,dim=-1)*one_hot
        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)

        pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)


        pos_neg_scores = p_model.calculate_logits_given_items(item_seq,item_seq_len,pos_neg_items, time_seq ,time_interval_seq ,target_time )
        targets = torch.zeros(batch_size,dtype=torch.long).to(p_model.device)

        loss = self.criterion(pos_neg_scores,targets)


        return loss


class AdverCETime(nn.Module):

    def __init__(self,K):
        super(AdverCETime,self).__init__()
        self.K = K
        self.type_criterion = nn.CrossEntropyLoss()
        self.time_criterion = nn.MSELoss()


    def forward(self,p_model,q_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time ):

        batch_size = item_seq.shape[0]

        noise_logits  = q_model.calculate_logits (item_seq, item_seq_len, time_seq,time_interval_seq,target_time)
        ones = torch.ones(noise_logits.shape).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        # one_hot_noise_logits = noise_logits * one_hot

        """
        Sampling
        """


        # 按照概率采样

        one_hot_noise_prob = F.softmax(noise_logits,dim=-1)*one_hot


        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)
        # print(neg_items)

        pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)

        pos_neg_scores,predict_intervals = p_model.calculate_logits_time_given_items(item_seq, item_seq_len, pos_neg_items, time_seq,
                                                              time_interval_seq, target_time)
        targets = torch.zeros(batch_size, dtype=torch.long).to(p_model.device)

        type_loss = self.type_criterion(pos_neg_scores, targets)

        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        target_interval = target_time.reshape(-1, 1) - last_time
        granularity = 24 * 30*6  # 对于亚马逊这个数据集以半年为单位
        time_loss = self.time_criterion(predict_intervals/granularity, target_interval/granularity)/5  # 用天做单位

        # granularity = 24 * 14  # 对于tmall这个数据集以周为单位
        # time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity)  # 用天做单位

        loss = type_loss + time_loss
        return loss
