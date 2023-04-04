# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from utils.tools import gather_indexes
import torch.nn.functional as F
class C_NCE(nn.Module):

    def __init__(self,K ,device):
        super(C_NCE,self).__init__()
        self.K = K
        self.device = device
        probs_data = torch.zeros(
            1, K, dtype=torch.float32, device=self.device)
        self.probs = nn.Parameter(probs_data)  # parameter probs
        self.criterion = nn.CrossEntropyLoss()


    def forward(self,p_model,q_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time):

        sampled_dt,sampled_type,mask_noise,accept_prob = q_model.sample_noises(item_seq,item_seq_len,time_seq,time_interval_seq,target_time,self.K,self.probs)


        target_time_interval = target_time.reshape(-1,1)-gather_indexes(time_seq.unsqueeze(2),item_seq_len-1) # B,1

        actual_sampled_dt = torch.cat((target_time_interval,sampled_dt),1) # B,1+K
        actual_sampled_type = torch.cat((target_id.reshape(-1,1),sampled_type),1) # B, 1+K


        noise_seq_output = q_model.forward(item_seq,item_seq_len)
        noise_prob = q_model.get_intensity_give_time_type(  actual_sampled_dt,actual_sampled_type,noise_seq_output)

        actual_seq_output = p_model.forward(item_seq,item_seq_len)
        actual_prob = p_model.get_intensity_give_time_type(actual_sampled_dt,actual_sampled_type,actual_seq_output)

        actual_noise_mask = torch.cat((torch.ones(target_time_interval.shape).to(self.device),mask_noise),1) # B, 1+K

        deno = self.K * noise_prob + actual_prob +1e-6  # B,(1+K)
        tmp1 = actual_prob / deno
        tmp2 = noise_prob / deno

        likeli = torch.cat((tmp1[:, 0].reshape(-1, 1), tmp2[:, 1:]), 1) +1e-6 # B, (1+K)
        # log_likeli = torch.sum(torch.log(likeli)*actual_noise_mask,dim=1)
        log_likeli = torch.sum(torch.log(likeli)  , dim=1)
        loss = -torch.mean(log_likeli)

        return loss


class NegNCE_f(nn.Module):

    def __init__(self, K):
        super(NegNCE_f, self).__init__()
        self.K = K
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, p_model, q_model, item_seq, item_seq_len, target_id, time_seq, time_interval_seq, target_time):
        # TODO 这个其实不是nce，只是用noise model选了一些样本，然后再用ce做的，需要改

        batch_size = item_seq.shape[0]
        # 使用q model计算neg items TODO 可优化，后续可以先计算好
        noise_logits = q_model.calculate_logits(item_seq, item_seq_len, time_seq=None,
                                                              time_interval_seq=None, target_time=None)
        ones = torch.ones(noise_logits.shape).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        one_hot_noise_logits = noise_logits * one_hot

        """
        Sampling
        """

        # 固定50个，随机100个

        # fixed_k = int(self.K / 3)
        #
        # _, neg_items = one_hot_noise_logits.topk(k=fixed_k)
        # random_neg_items = p_model.sample_negs(batch_size, self.K)  # B, random_k
        # neg_items = torch.cat((neg_items, random_neg_items), 1)
        #
        # zeros = torch.zeros(noise_logits.shape).to(p_model.device)
        # one_hot = zeros.scatter_(1, neg_items, 1)  # 所有采样的neg 置1
        # one_hot = one_hot.scatter_(1, target_id.reshape(-1, 1), 0)  # 所有的target id置0
        #
        # _, neg_items = one_hot.topk(k=self.K)
        #
        # pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)  # B,1+K

        # 按照概率采样

        softmax_fun = nn.Softmax(-1)
        one_hot_noise_prob = softmax_fun(one_hot_noise_logits)*one_hot

        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)

        pos_neg_items = torch.cat((target_id.reshape(-1,1),neg_items),1)




        """
        Scoring
        """
        pos_neg_scores = p_model.calculate_logits_given_items(item_seq, item_seq_len, pos_neg_items, time_seq=None,
                                                              time_interval_seq=None, target_time=None)

        targets = torch.zeros(batch_size, dtype=torch.long).to(p_model.device)
        loss = self.criterion(pos_neg_scores, targets)



        return loss

class NegNCE(nn.Module):

    def __init__(self,K):
        super(NegNCE,self).__init__()
        self.K = K

    def forward(self,p_model,q_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time ):


        # pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)
        # sample_emb = q_model.item_embedding(pos_neg_items)  # B, 1+K, emb_size
        # noise_prob = torch.sum(seq_emb.unsqueeze(1) * sample_emb, 2)  # B, 1+K
        noise_logits = q_model.calculate_logits(item_seq, item_seq_len, time_seq, time_interval_seq, target_time)
        ones = torch.ones(noise_logits.shape).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        one_hot_noise_prob = F.softmax(noise_logits, dim=-1) * one_hot
        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)
        pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)
        """
        Scoring
        """
        noise_prob = torch.gather(noise_logits, 1, pos_neg_items)


        actual_prob = p_model.calculate_logits_given_items( item_seq,item_seq_len,pos_neg_items, time_seq ,time_interval_seq ,target_time )

        # NCE 1
        # relu_fun = nn.ReLU()
        # noise_prob = relu_fun(noise_prob)+1e-6
        # actual_prob = relu_fun(actual_prob) +1e-6

        softmax_fun = nn.Softmax(-1)
        noise_prob = softmax_fun(noise_prob)
        actual_prob = softmax_fun(actual_prob)

        deno = self.K * noise_prob + actual_prob + 1e-6  # 如果把self.K * 删去呢？
        tmp1 = actual_prob / deno
        tmp2 = noise_prob / deno

        likeli = torch.cat((tmp1[:, 0].reshape(-1, 1), tmp2[:, 1:]), 1)
        likeli[likeli==1] = 1+1e-6 # 避免反向传播出现nan



        log_likeli = torch.log(likeli )



        loss = -torch.mean(log_likeli)

        return loss

class NegNCETime(nn.Module):

    def __init__(self,K):
        super(NegNCETime,self).__init__()
        self.K = K
        self.time_criterion = nn.MSELoss()

    def forward(self,p_model,q_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time ):


        # pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)
        #
        # sample_emb = q_model.item_embedding(pos_neg_items)  # B, 1+K, emb_size
        #
        # noise_prob = torch.sum(seq_emb.unsqueeze(1) * sample_emb, 2)  # B, 1+K

        noise_logits = q_model.calculate_logits(item_seq, item_seq_len, time_seq, time_interval_seq, target_time)
        ones = torch.ones(noise_logits.shape).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        one_hot_noise_prob = F.softmax(noise_logits, dim=-1) * one_hot
        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)
        pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)
        """
        Scoring
        """
        noise_prob = torch.gather(noise_logits, 1, pos_neg_items)




        actual_prob,predict_intervals = p_model.calculate_logits_time_given_items( item_seq,item_seq_len,pos_neg_items, time_seq ,time_interval_seq ,target_time )



        # NCE 1
        # relu_fun = nn.ReLU()
        # noise_prob = relu_fun(noise_prob)+1e-6
        # actual_prob = relu_fun(actual_prob) +1e-6

        softmax_fun = nn.Softmax(-1)
        noise_prob = softmax_fun(noise_prob)
        actual_prob = softmax_fun(actual_prob)

        deno = self.K * noise_prob + actual_prob + 1e-6  # 如果把self.K * 删去呢？
        tmp1 = actual_prob / deno
        tmp2 = noise_prob / deno

        likeli = torch.cat((tmp1[:, 0].reshape(-1, 1), tmp2[:, 1:]), 1)
        # likeli[likeli==1] = 1+1e-6 # 避免反向传播出现nan

        log_likeli = torch.log(likeli )

        type_loss = -torch.mean(log_likeli)

        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        target_interval = target_time.reshape(-1, 1) - last_time
        granularity = 24 * 30 * 6  # 对于亚马逊这个数据集以半年为单位
        time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity) / 5  # 用天做单位

        # granularity = 24 * 14  # 对于tmall这个数据集以周为单位
        # time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity)  # 用天做单位

        loss = type_loss +time_loss

        return loss

class RandomNCE(nn.Module):

    def __init__(self,K):
        super(RandomNCE,self).__init__()
        self.K = K

    def forward(self,p_model ,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time ):

        batch_size = item_seq.shape[0]

        ones = torch.ones(batch_size, p_model.n_items).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        # one_hot_noise_logits = noise_logits * one_hot
        one_hot_noise_prob = F.softmax(one_hot, dim=-1) * one_hot
        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)


        pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)


        """
        Scoring
        """
        noise_prob = torch.rand(batch_size,self.K+1).to(target_id.device)
        actual_prob = p_model.calculate_logits_given_items( item_seq,item_seq_len,pos_neg_items, time_seq,time_interval_seq,target_time)

        # NCE 1
        # relu_fun = nn.ReLU()
        # noise_prob = relu_fun(noise_prob)+1e-6
        # actual_prob = relu_fun(actual_prob) +1e-6

        softmax_fun = nn.Softmax(-1)
        noise_prob = softmax_fun(noise_prob)
        actual_prob = softmax_fun(actual_prob)

        deno = self.K * noise_prob + actual_prob +1e-6 # 如果把self.K * 删去呢？
        tmp1 = actual_prob / deno
        tmp2 = noise_prob / deno

        likeli = torch.cat((tmp1[:, 0].reshape(-1, 1), tmp2[:, 1:]), 1)

        # tmp =(likeli == 1).long()
        # sum_val = torch.sum(tmp)
        # if sum_val.data>0:
        #     print(sum_val)

        log_likeli = torch.log(likeli )
        loss = -torch.mean(log_likeli)

        return loss

class RandomNCETime(nn.Module):

    def __init__(self,K):
        super(RandomNCETime,self).__init__()
        self.K = K
        self.time_criterion = nn.MSELoss()

    def forward(self,p_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time ):

        batch_size = item_seq.shape[0]

        ones = torch.ones(batch_size, p_model.n_items).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        # one_hot_noise_logits = noise_logits * one_hot
        one_hot_noise_prob = F.softmax(one_hot, dim=-1) * one_hot
        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)

        pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)

        """
        Scoring
        """
        noise_prob = torch.rand(batch_size,self.K+1).to(target_id.device)
        actual_prob,predict_intervals = p_model.calculate_logits_time_given_items( item_seq,item_seq_len,pos_neg_items, time_seq,time_interval_seq,target_time)

        # NCE 1
        # relu_fun = nn.ReLU()
        # noise_prob = relu_fun(noise_prob)+1e-6
        # actual_prob = relu_fun(actual_prob) +1e-6

        softmax_fun = nn.Softmax(-1)
        noise_prob = softmax_fun(noise_prob)
        actual_prob = softmax_fun(actual_prob)

        deno = self.K * noise_prob + actual_prob +1e-6 # 如果把self.K * 删去呢？
        tmp1 = actual_prob / deno
        tmp2 = noise_prob / deno

        likeli = torch.cat((tmp1[:, 0].reshape(-1, 1), tmp2[:, 1:]), 1)

        # tmp =(likeli == 1).long()
        # sum_val = torch.sum(tmp)
        # if sum_val.data>0:
        #     print(sum_val)

        log_likeli = torch.log(likeli )
        type_loss = -torch.mean(log_likeli)

        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        target_interval = target_time.reshape(-1, 1) - last_time
        granularity = 24 * 30 * 6  # 对于亚马逊这个数据集以半年为单位
        time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity) / 5  # 用天做单位

        # granularity = 24 * 14  # 对于tmall这个数据集以周为单位
        # time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity)  # 用天做单位

        loss = type_loss + time_loss

        return loss

class AdverNCE(nn.Module):

    def __init__(self,K):
        super(AdverNCE,self).__init__()
        self.K = K

    def forward(self,p_model,q_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time ):

        batch_size = item_seq.shape[0]

        noise_logits  = q_model.calculate_logits (item_seq, item_seq_len, time_seq,time_interval_seq,target_time)
        ones = torch.ones(noise_logits.shape).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        one_hot_noise_prob = F.softmax(noise_logits,dim=-1)*one_hot
        neg_items = torch.multinomial(one_hot_noise_prob, self.K, replacement=False)

        pos_neg_items = torch.cat((target_id.reshape(-1, 1), neg_items), 1)


        """
        Scoring
        """
        noise_prob = torch.gather(noise_logits,1,pos_neg_items)
        actual_prob = p_model.calculate_logits_given_items( item_seq,item_seq_len,pos_neg_items, time_seq,time_interval_seq,target_time)

        # NCE 1
        # relu_fun = nn.ReLU()
        # noise_prob = relu_fun(noise_prob)+1e-6
        # actual_prob = relu_fun(actual_prob) +1e-6

        softmax_fun = nn.Softmax(-1)
        noise_prob = softmax_fun(noise_prob)
        actual_prob = softmax_fun(actual_prob)

        deno = self.K * noise_prob + actual_prob +1e-6 # 如果把self.K * 删去呢？
        tmp1 = actual_prob / deno
        tmp2 = noise_prob / deno

        likeli = torch.cat((tmp1[:, 0].reshape(-1, 1), tmp2[:, 1:]), 1)

        # tmp =(likeli == 1).long()
        # sum_val = torch.sum(tmp)
        # if sum_val.data>0:
        #     print(sum_val)

        log_likeli = torch.log(likeli )
        loss = -torch.mean(log_likeli)

        return loss


class AdverNCETime(nn.Module):

    def __init__(self,K):
        super(AdverNCETime,self).__init__()
        self.K = K
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

        """
        Scoring
        """
        noise_prob = torch.gather(noise_logits,1,pos_neg_items)
        actual_prob,predict_intervals = p_model.calculate_logits_time_given_items( item_seq,item_seq_len,pos_neg_items, time_seq,time_interval_seq,target_time)

        # NCE 1
        # relu_fun = nn.ReLU()
        # noise_prob = relu_fun(noise_prob)+1e-6
        # actual_prob = relu_fun(actual_prob) +1e-6

        softmax_fun = nn.Softmax(-1)
        noise_prob = softmax_fun(noise_prob)
        actual_prob = softmax_fun(actual_prob)


        deno = self.K * noise_prob + actual_prob +1e-6 # 如果把self.K * 删去呢？
        tmp1 = actual_prob / deno
        tmp2 = noise_prob / deno

        likeli = torch.cat((tmp1[:, 0].reshape(-1, 1), tmp2[:, 1:]), 1)
        # tmp =(likeli == 1).long()
        # sum_val = torch.sum(tmp)
        # if sum_val.data>0:
        #     print(sum_val)

        log_likeli = torch.log(likeli )
        type_loss = -torch.mean(log_likeli)

        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        target_interval = target_time.reshape(-1, 1) - last_time
        granularity = 24 * 30 * 6  # 对于亚马逊这个数据集以半年为单位
        time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity) / 5  # 用天做单位

        # granularity = 24 * 14  # 对于tmall这个数据集以周为单位
        # time_loss = self.time_criterion(predict_intervals / granularity, target_interval / granularity)  # 用天做单位

        loss = type_loss + time_loss

        return loss

class NCE(nn.Module):

    def __init__(self,K):
        super(NCE,self).__init__()
        self.K = K

    def forward(self,p_model,q_model,item_seq,item_seq_len,target_id):

        # noise_score,noise_pack  = q_model.get_noise_score(item_seq,item_seq_len,target_id,self.K)
        #
        # target_score = p_model.get_target_score(item_seq,item_seq_len,target_id)
        #
        # loss =  target_score / (torch.sum(noise_score, axis=1) + target_score)
        #
        # loss = -torch.sum(loss)

        relu_fun = nn.ReLU()
        device = item_seq.device

        noise_logits = q_model.calculate_logits(item_seq,item_seq_len)
        actual_logits = p_model.calculate_logits(item_seq,item_seq_len)

        ones = torch.ones(noise_logits.shape).to(device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)
        one_hot_noise_logits = noise_logits*one_hot

        _,topk_indices = one_hot_noise_logits.topk(k=self.K)

        actual_noise_indices = torch.cat((target_id.reshape(-1,1),topk_indices),1) # B*(K+1)
        noise_prob = torch.gather(noise_logits,1,actual_noise_indices)
        actual_prob = torch.gather(actual_logits,1,actual_noise_indices)

        # NCE 1

        softmax_fun = nn.Softmax(-1)
        noise_prob = softmax_fun(noise_prob)
        actual_prob = softmax_fun(actual_prob)

        deno =  self.K*noise_prob + actual_prob # 如果把self.K * 删去呢？
        tmp1 = actual_prob/deno
        tmp2 = noise_prob/deno

        likeli = torch.cat((tmp1[:,0].reshape(-1,1),tmp2[:,1:]),1)
        log_likeli = torch.log(likeli)
        loss = -torch.mean(log_likeli )

        # NCE 2

        # actual_noise_logits = actual_prob
        # labels = torch.ones(actual_noise_logits.shape[0],actual_noise_logits.shape[1]).to(actual_noise_logits.device)
        # labels[:,1:] = 0
        # sigmoid_fun = torch.nn.Sigmoid()
        # sigmoid_actual_noise_logits = sigmoid_fun(actual_noise_logits) # B*[k+1]
        #
        # loss = -(torch.log(sigmoid_actual_noise_logits) * labels + torch.log(1-sigmoid_actual_noise_logits)*(1-labels))
        # loss = torch.mean(loss)


        return loss





