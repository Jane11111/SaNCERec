# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset


class PaddedDatasetTrain(Dataset):
    def __init__(self, sessions, max_len):
        self.sessions = sessions
        self.max_len = max_len


    def __getitem__(self, idx):
        seq = self.sessions[idx]


        # for seq in seqs:
        #     [user_id,item_lst, target_id, length, u_A_in, u_A_out]
        user_id = seq[0]
        item_lst = seq[1]
        target_id = seq[2]
        length = seq[3]
        time_lst = seq[4]
        time_interval_lst = seq[5]
        target_time = seq[6]
        seq_emb = seq[7]
        sample_items = seq[8]


        seq_padding_size = [0, self.max_len-length]

        padded_item_lst = np.pad(item_lst, seq_padding_size,'constant')
        padded_time_lst = np.pad(time_lst,seq_padding_size,'constant')
        padded_time_interval_lst = np.pad(time_interval_lst,seq_padding_size,'constant')



        return  [user_id, padded_item_lst, target_id, length,padded_time_lst,padded_time_interval_lst,target_time,seq_emb,sample_items ]
        # return seq

    def __len__(self):
        return len(self.sessions)

class PaddedDatasetEvaluate(Dataset):
    def __init__(self, sessions, max_len):
        self.sessions = sessions
        self.max_len = max_len


    def __getitem__(self, idx):
        seq = self.sessions[idx]


        # for seq in seqs:
        #     [user_id,item_lst, target_id, length, u_A_in, u_A_out]
        user_id = seq[0]
        item_lst = seq[1]
        target_id = seq[2]
        length = seq[3]
        time_lst = seq[4]
        time_interval_lst = seq[5]
        target_time = seq[6]
        # history_dates = seq[7]
        # target_dates = [seq[8][1],seq[8][3]]

        # print(length)
        seq_padding_size = [0, self.max_len-length]

        padded_item_lst = np.pad(item_lst, seq_padding_size,'constant')
        padded_time_lst = np.pad(time_lst,seq_padding_size,'constant')
        padded_time_interval_lst = np.pad(time_interval_lst,seq_padding_size,'constant')



        # # history_years = np.pad(history_dates[0],seq_padding_size,'constant')
        # history_month = np.pad(history_dates[1],seq_padding_size,'constant')
        # # history_day = np.pad(history_dates[2], seq_padding_size, 'constant')
        # history_dayofweek = np.pad(history_dates[3], seq_padding_size, 'constant')
        # # history_hour = np.pad(history_dates[4], seq_padding_size, 'constant')
        # padded_history_dates = [history_month,history_dayofweek]

        return  [user_id, padded_item_lst, target_id, length,padded_time_lst,padded_time_interval_lst,target_time ]
        # return seq

    def __len__(self):
        return len(self.sessions)
