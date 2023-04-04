# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle

class PrepareData():
    def __init__(self,config,logger):
        
        root = ''


        self.normal_train_path = root + 'training_testing_data/'+config[
            'dataset'] + '_time_item_based_unidirection/train_data_time_loo.pt'
        self.normal_test_path = root + 'training_testing_data/'+config[
            'dataset']+'_time_item_based_unidirection/test_data_time_loo.pt'
        self.normal_dev_path = root + 'training_testing_data/'+config[
            'dataset']+'_time_item_based_unidirection/dev_data_time_loo.pt'


        self.origin_path = root + 'orgin_data/'+config['dataset']+'.csv'

        self.config=config
        self.logger = logger

        self.train_limit = 10000000
        self.test_limit = 2000000
        self.dev_limit = 2000000

        # self.train_limit = 10000
        # self.test_limit = 20
        # self.dev_limit = 20


    def get_statistics(self):

        df = pd.read_csv(self.origin_path)

        user_set = set(df['user_id'].tolist())
        self.logger.info('the user count is: %d'%(len(user_set)))
        item_set = set(df['item_id'].tolist())
        self.logger.info('the item count is: %d'%(len(item_set)))
        behavior_count = df.shape[0]
        self.logger.info('the behavior count is : %d'%(behavior_count))

        behavior_per_user = df.groupby(by=['user_id'], as_index=False)['item_id'].count()
        behavior_per_user = behavior_per_user['item_id'].mean()
        self.logger.info('the avg behavior of each user count is: %.5f'%(behavior_per_user))

        behavior_per_item = df.groupby(by=['item_id'], as_index=False)['user_id'].count()
        behavior_per_item = behavior_per_item['user_id'].mean()
        self.logger.info('the avg behavior of each item count is: %.5f'%(behavior_per_item))

        self.user_count = len(user_set)
        self.item_count = len(item_set)

    def get_item_num(self):
        return self.item_count



    def load_dataset_dev(self ):

        file = open(self.normal_dev_path, 'rb')
        dataset = pickle.loads(file.read())

        # TODO 时间归一化
        # dataset = self.normalize_all_time(dataset)

        limit = min(self.dev_limit, len(dataset))
        dataset = dataset[:limit]

        self.logger.info('test_length: %d' % (len(dataset)))

        return dataset

    def load_dataset_test(self ):

        file = open(self.normal_test_path, 'rb')
        dataset = pickle.loads(file.read())

        # TODO 时间归一化
        # dataset = self.normalize_all_time(dataset)

        limit = min(self.test_limit, len(dataset))
        dataset = dataset[:limit]

        self.logger.info('test_length: %d' % (len( dataset)))

        return dataset

    def load_dataset_train_simple(self ):



        file = open(self.normal_train_path, 'rb')
        dataset = pickle.loads(file.read())

        # TODO 时间归一化
        # dataset = self.normalize_all_time(dataset)

        limit = min(self.train_limit, len(dataset))
        dataset = dataset[:limit]
        self.logger.info('train_length: %d' % (len(dataset)))
        return dataset

    # def load_dataset_train(self ):
    #
    #     # TODO 需要加上load负样本的部分
    #
    #     file = open(self.normal_train_path, 'rb')
    #     dataset = pickle.loads(file.read())
    #
    #     # TODO 时间归一化
    #     # dataset = self.normalize_all_time(dataset)
    #
    #     limit = min(self.train_limit, len(dataset))
    #     dataset = dataset[:limit]
    #
    #     # if (self.config['train_method'] == 'neg_nce' or self.config['train_method'] == 'neg_ce'):
    #     emb_f = open(self.config['noise_emb_path'],'rb')
    #     emb_lst = pickle.loads(emb_f.read())
    #     # emb_lst = np.array(emb_lst)
    #
    #     sample_f = open(self.config['noise_sample_path'],'rb')
    #     sample_lst = pickle.loads(sample_f.read())
    #     # sample_lst = np.array(sample_lst)
    #
    #     count = 0
    #     for i in range(len(dataset)):
    #         dataset[i] = list(dataset[i])
    #         dataset[i].append(emb_lst[i])
    #
    #         for j in range(len(sample_lst[i])):
    #             item = sample_lst[i][j]
    #
    #             if item>=self.item_count:
    #                 sample_lst[i][j] = 0
    #                 count+=1
    #         dataset[i].append(sample_lst[i])
    #     print(count)
    #     print('----------------------------------------')
    #
    #
    #     dataset = dataset[:limit]
    #     # dataset = dataset[-limit:]
    #     self.logger.info('train_length: %d' % (len(dataset)))
    #     return dataset
