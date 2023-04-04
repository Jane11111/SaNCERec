# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F

class MaximumLogLikelihood(nn.Module):

    def __init__(self ):
        super(MaximumLogLikelihood,self).__init__()
        # self.K = K

    def forward(self,p_model,item_seq,item_seq_len,target_id,time_seq,time_interval_seq,target_time ):

        # 如果uniform采10个数然后再缩放呢  这10个数不包括target time


        batch_size = item_seq.shape[0]
        relu_fun = torch.nn.ReLU()

        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1).reshape(-1,)

        k = 10
        sampled_nums = torch.rand((1,k)).to(item_seq.device)

        sampled_times = last_time.reshape(-1,1) + (target_time-last_time).reshape(-1,1)*sampled_nums # batchsize, sample_number


        neg_term = torch.zeros((batch_size,)).to(item_seq.device)
        for i in range(k):

            cur_target_time = sampled_times[:,i]

            cur_logits = p_model.calculate_logits(item_seq, item_seq_len, time_seq,time_interval_seq,cur_target_time)
            cur_logits = relu_fun(cur_logits)

            neg_term += torch.sum(cur_logits,1) # 负项相加

        target_logits = p_model.calculate_logits(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)
        target_logits = relu_fun(target_logits)

        ones = torch.zeros(target_logits.shape).to(p_model.device)
        one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 1)
        target_logits *= one_hot

        target_logits = torch.sum(target_logits,1)


        log_likeli = torch.log(target_logits+1) -neg_term


        loss = -torch.mean(log_likeli)

        return loss