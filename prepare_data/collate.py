# -*- coding: utf-8 -*-

import torch
import numpy as np
import dgl
from collections import Counter




def collate_fn_train(samples):
    # samples = np.array(samples)

    # TODO 需要处理负采样样本
    user_id, item_seq, target_id, item_seq_len,time_seq ,time_interval_seq,target_time,seq_emb,sample_items  = zip(*samples)

    user_id = torch.tensor(user_id,dtype = torch.long)
    item_seq = torch.tensor(item_seq, dtype=torch.long)
    target_id = torch.tensor(target_id, dtype=torch.long)
    item_seq_len = torch.tensor(item_seq_len, dtype=torch.long)
    time_seq = torch.tensor(time_seq ,dtype = torch.float)
    time_interval_seq = torch.tensor(time_interval_seq,dtype = torch.float )
    target_time = torch.tensor(target_time,dtype = torch.float )
    seq_emb = torch.tensor(seq_emb,dtype = torch.float)
    sample_items = torch.tensor(sample_items,dtype = torch.long)

    return user_id, item_seq, target_id, item_seq_len,time_seq,time_interval_seq,target_time, seq_emb, sample_items


def collate_fn_evaluate(samples):
    # samples = np.array(samples)
    user_id, item_seq, target_id, item_seq_len,time_seq ,time_interval_seq,target_time = zip(*samples)

    user_id = torch.tensor(user_id,dtype = torch.long)
    item_seq = torch.tensor(item_seq, dtype=torch.long)
    target_id = torch.tensor(target_id, dtype=torch.long)
    item_seq_len = torch.tensor(item_seq_len, dtype=torch.long)
    time_seq = torch.tensor(time_seq ,dtype = torch.float)
    time_interval_seq = torch.tensor(time_interval_seq,dtype = torch.float )
    target_time = torch.tensor(target_time,dtype = torch.float )

    # history_dates = torch.tensor(history_dates,dtype = torch.long)
    # target_dates = torch.tensor(target_dates,dtype = torch.long)

    return user_id, item_seq, target_id, item_seq_len,time_seq,time_interval_seq,target_time
