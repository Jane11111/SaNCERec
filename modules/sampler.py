# -*- coding: utf-8 -*-

import json
import torch
import torch.nn as nn
import pickle
import logging
from logging import getLogger
from model.sasrec import SASRec
from model.t_sasrec import TimeAwareSASRec
# from utils.trainer import TrainRunnerNormal
from torch.utils.data import DataLoader
from prepare_data.preprocess import PrepareData
from prepare_data.dataset import PaddedDatasetTrain,PaddedDatasetEvaluate
from prepare_data.collate import  collate_fn_evaluate


class RandomSampler:

    def __init__(self, dataset ,emb_save_path,sample_save_path,logger,sample_round,K,device = 0):

        # self.model_name = model_name
        # self.model_path = model_path
        self.config = {'dataset':dataset}
        self.emb_save_path = emb_save_path
        self.sample_save_path = sample_save_path
        self.sample_round = sample_round
        self.K = K
        self.logger = logger
        self.device = 'cuda:'+str(device)

        self.prepare_data_model = PrepareData(self.config, self.logger)
        self.prepare_data_model.get_statistics()
        self.num_items = self.prepare_data_model.get_item_num()


    def load_dataset(self):

        test_sessions = self.prepare_data_model.load_dataset_train_simple()

        test_set = PaddedDatasetEvaluate(test_sessions, 50)

        test_loader = DataLoader(
            test_set,
            batch_size=512,
            shuffle=False,
            collate_fn=collate_fn_evaluate,
        )

        return test_loader

    def save(self,emb_lst,samples):
        emb_lst = emb_lst.cpu().numpy().tolist()
        samples = samples.cpu().numpy().tolist()

        fp = open(self.emb_save_path, 'wb')
        fp.write(pickle.dumps(emb_lst))
        fp.close()

        fp = open(self.sample_save_path, 'wb')
        fp.write(pickle.dumps(samples))
        fp.close()




    def random_sample(self):

        test_loader = self.load_dataset()


        # def prepare_batch(inputs, device):
        #     inputs_gpu = [x.to(device) for x in inputs]
        #     return inputs_gpu


        emb_lst  = torch.Tensor().to(self.device)
        samples = torch.Tensor().long().to(self.device)
        count = 0
        with torch.no_grad():
            for batch in test_loader:
                user_id, item_seq, target_id, item_seq_len, time_seq, time_seq_len, target_time = batch

                target_id = target_id.to(self.device)

                seq_emb = torch.ones((item_seq.shape[0],1)).to(self.device)
                count += item_seq_len.shape[0]



                ones = torch.ones((item_seq.shape[0],self.num_items)).to(self.device)
                one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)

                # one_hot = torch.arange(0,self.num_items).reshape(1,-1).repeat(item_seq.shape[0],1).float()


                # softmax_fun = nn.Softmax(-1)
                # one_hot_noise_prob = softmax_fun(one_hot) * one_hot


                cur_samples = torch.multinomial(one_hot, self.K*self.sample_round, replacement=False)

                    # neg_items[neg_items >= self.config['num_items']] = 0

                    # cur_samples = torch.cat((cur_samples,neg_items),1)

                emb_lst = torch.cat((emb_lst, seq_emb), 0)
                samples = torch.cat((samples,cur_samples),0)
                print(count)


        self.save(emb_lst,samples)
        return emb_lst,samples


class MultinomialSampler:

    def __init__(self,model_name,model_path, config_path ,emb_save_path,sample_save_path,logger,sample_round,K):

        self.model_name = model_name
        self.model_path = model_path
        self.config_path = config_path
        self.emb_save_path = emb_save_path
        self.sample_save_path = sample_save_path
        self.sample_round = sample_round
        self.K = K
        self.logger = logger
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)


    def load_model(self):


        loaded_model_dict = torch.load(self.model_path)

        if self.model_name == 'SASRec':
            model_obj = SASRec(self.config,self.config['num_items'])
        elif self.model_name == 'TimeAwareSASRec':
            model_obj = TimeAwareSASRec(self.config,self.config['num_items'])
        model_obj.load_state_dict(loaded_model_dict)

        return model_obj

    def load_dataset(self):
        prepare_data_model = PrepareData(self.config,self.logger)
        test_sessions = prepare_data_model.load_dataset_train_simple()

        test_set = PaddedDatasetEvaluate(test_sessions, self.config['max_len'])

        test_loader = DataLoader(
            test_set,
            batch_size=self.config['train_batch_size'],
            shuffle=False,
            collate_fn=collate_fn_evaluate,
        )

        return test_loader

    def save(self,emb_lst,samples):
        emb_lst = emb_lst.cpu().numpy().tolist()
        samples = samples.cpu().numpy().tolist()

        fp = open(self.emb_save_path, 'wb')
        fp.write(pickle.dumps(emb_lst))
        fp.close()

        fp = open(self.sample_save_path, 'wb')
        fp.write(pickle.dumps(samples))
        fp.close()




    def random_sample(self):

        model_obj = self.load_model().to(self.config['device'])
        test_loader = self.load_dataset()

        model_obj.eval()

        def prepare_batch(inputs, device):
            inputs_gpu = [x.to(device) for x in inputs]
            return inputs_gpu

        emb_lst  = torch.Tensor().to(self.config['device'])
        samples = torch.Tensor( ).long().to(self.config['device'])

        count = 0
        with torch.no_grad():
            for batch in test_loader:
                user_id, item_seq, target_id, item_seq_len, time_seq, time_seq_len, target_time = prepare_batch(batch,self.config['device'])
                seq_emb,logits = model_obj.get_emb_logits(item_seq, item_seq_len, time_seq, time_seq_len, target_time)

                count += seq_emb.shape[0]
                emb_lst = torch.cat((emb_lst,seq_emb),0)

                ones = torch.ones(logits.shape).to(self.config['device'])
                one_hot = ones.scatter_(1, target_id.reshape(-1, 1), 0)

                softmax_fun = nn.Softmax(-1)
                one_hot_noise_prob = softmax_fun(logits) * one_hot

                # cur_samples = torch.Tensor().long().to(self.config['device'])
                # for i in range(self.sample_round):
                cur_samples = torch.multinomial(one_hot_noise_prob, self.K*sample_round, replacement=False)

                    # neg_items[neg_items>=self.config['num_items']] = 0
                    #
                    # tmp = (neg_items>=self.config['num_items']).long()
                    # sum_val = torch.sum(tmp)
                    # if sum_val.data>0:
                    #     print(sum_val)

                    # cur_samples = torch.cat((cur_samples,neg_items),1)
                samples = torch.cat((samples,cur_samples),0)
                print(count)
        self.save(emb_lst,samples)
        return emb_lst,samples



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = getLogger('test')
    handler = logging.FileHandler('test_log.txt')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s   %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    model_name = 'SASRec'
    data_name = 'elec'
    sample_method = 'random'
    root = '/home/zxl/project/NCERec/data/'

    emb_save_path = root + 'samples/'+data_name+'_emb_'+sample_method+'.pkl'
    sample_save_path= root + 'samples/'+data_name+'_samples_'+sample_method+'.pkl'
    sample_round = 3
    K = 30



    sample_model = RandomSampler(data_name,emb_save_path,sample_save_path,logger, sample_round,K,device = 0)
    sample_model.random_sample()

    # saved_name = 'SASRec_elec_2021-11-02_11-52-50_neg_ce_0.0005'
    # model_path = root + 'model/' + saved_name + '.pkl'
    # config_path = root + 'model/' + saved_name + '_config.json'
    # sampler = MultinomialSampler(model_name,model_path,config_path,emb_save_path,sample_save_path,logger,sample_round,K)
    # sampler.random_sample()
    #

    print('aaaa')






