# -*- coding: utf-8 -*-



import warnings
import json
import copy

warnings.filterwarnings('ignore')
import traceback
import torch as th
from torch.utils.data import DataLoader
from prepare_data.preprocess import PrepareData
from prepare_data.dataset import PaddedDatasetTrain,PaddedDatasetEvaluate
from prepare_data.collate import collate_fn_train, collate_fn_evaluate
from utils.trainer import TrainRunnerTime

from modules.sampler import RandomSampler,MultinomialSampler
from model.sasrec import SASRec
import yaml
import logging
from logging import getLogger

import time
import os
import numpy as np
import random
import json
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def seed_torch(seed=2020):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	th.manual_seed(seed)
	th.cuda.manual_seed(seed)
	th.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	th.backends.cudnn.benchmark = False
	th.backends.cudnn.deterministic = True

seed_torch()

def load_data(prepare_data_model,config):
     
    train_sessions = prepare_data_model.load_dataset_train_simple()
    train_set = PaddedDatasetEvaluate(train_sessions, max_len)
    train_f = collate_fn_evaluate

    test_sessions = prepare_data_model.load_dataset_test()
    dev_sessions = prepare_data_model.load_dataset_dev()
    test_set = PaddedDatasetEvaluate(test_sessions, max_len)
    dev_set = PaddedDatasetEvaluate(dev_sessions, max_len)
    train_loader = DataLoader(
        train_set,
        batch_size=config['train_batch_size'],
        shuffle=True,
        drop_last=False,
        collate_fn=train_f,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        collate_fn=collate_fn_evaluate,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config['train_batch_size'],
        shuffle=False,
        collate_fn=collate_fn_evaluate,
    )

    return train_loader, test_loader, dev_loader


def load_hyper_param(config, model,data_name=None):


    res = []

    if 'SASRec' in model:
        # for learning_rate in [  0.001,0.0005]:
        for learning_rate in [0.001]:
            for n_layers in [1 ]:
                for n_heads in [ 1]:
                    for hidden_dropout_prob in [0]:
                        for attn_dropout_prob in [ 0]:
                            cur_config = config.copy()
                            cur_config['learning_rate'] = learning_rate
                            cur_config['n_layers'] = n_layers
                            cur_config['n_heads'] = n_heads
                            cur_config['hidden_dropout_prob'] = hidden_dropout_prob
                            cur_config['attn_dropout_prob'] = attn_dropout_prob
                            cur_config['hyper_combi'] = str(learning_rate)+'_'+str(n_layers)+'_'+\
                                                        str(n_heads)+'_'+str(hidden_dropout_prob)+'_'+\
                                                        str(attn_dropout_prob)

                            res.append(cur_config)
    # elif 'GRU4Rec' in model or 'NARM' in model or 'STAMP' in model:
    for learning_rate in [0.001]:
    # for learning_rate in [0.001]:
        # for dropout_prob in [0,0.25,0.5]:
        cur_config = config.copy()
        cur_config['learning_rate'] = learning_rate
        # cur_config['dropout_prob'] = dropout_prob
        cur_config['hyper_combi'] = str(learning_rate)
        res.append(cur_config)

    return res
def load_logger(config):
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_path = './data/log/' + config['model'] + '_' + config['dataset'] + '_' + config['train_method'] + '_' + str(
        cur_time) + "_log.txt"
    log_name = '' + config['model'] + '_' + config['dataset'] + '_' + config['train_method'] + '_' + str(
        cur_time)+'/'

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = getLogger('test')
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s   %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(log_path)
    logger.info(config)
    return cur_time,log_name,logger
def load_config():
    root = '/home/zxl/project/NCERec/data/'
    # best_model_saved_name = 'GRU4Rec_elec_2021-11-09_10-02-52_neg_ce_0.005'
    # best_model_saved_name = 'GRU4Rec_movie_tv_2021-11-13_16-57-42_neg_ce_0.005'
    # best_model_saved_name = 'GRU4Rec_home_2021-11-13_16-52-37_neg_ce_0.005'
    # best_model_saved_name = 'GRU4Rec_sports_2021-11-13_17-07-32_neg_ce_0.005'
    # best_model_saved_name = 'GRU4Rec_movielen_2021-11-16_19-13-58_neg_ce_0.001'
    best_model_saved_name = 'SASRec_elec_2021-11-06_21-36-26_neg_ce_0.0005'
    # best_model_saved_name = 'SASRec_movie_tv_2021-11-08_19-33-30_neg_ce_0.0005'
    # best_model_saved_name = 'SASRec_home_2021-11-04_16-36-29_neg_ce_0.0005'
    # best_model_saved_name = 'SASRec_sports_2021-11-06_21-47-54_neg_ce_0.0005'
    # best_model_saved_name = 'SASRec_movielen_2021-11-07_19-53-47_neg_ce_0.001'

    best_model_path = root + 'model/' + best_model_saved_name + '.pkl'
    best_config_path = root + 'model/' + best_model_saved_name + '_config.json'
 
    config = {'model': model,
              'dataset': dataset,
              'train_method': train_method,
              'K': K,
              'sample_round': sample_round,
              'gpu_id': gpu_id,
              'epochs': epochs,
              'max_len': 50,
              'train_batch_size': train_batch_size,
              'device': 'cuda:' + str(gpu_id),
              # 'device':'cpu',
              }
    if train_method == 'neg_nce' or train_method == 'neg_nce_time' \
            or train_method == 'neg_ce' or train_method == 'neg_ce_time': # 模型以及采样样本保存的路径

        # config['noise_sample_path']= sample_save_path
        # config['noise_emb_path'] = emb_save_path
        # if train_method == 'neg_nce' or train_method == 'neg_nce_time' :
        config['noise_model_path'] = best_model_path
        config['noise_config_path']= best_config_path

    # if train_method == 'neg_nce' or train_method == 'neg_nce_time': #TODO gpu id 需要和neg_ce保持一致
        with open(best_config_path, 'r') as f:
            noise_config = json.load(f)
        config['K'] = noise_config['K']
        config['gpu_id'] = noise_config['gpu_id']
        config['sample_round'] = noise_config['sample_round']
        config['device'] = 'cuda:' + str(noise_config['gpu_id'])

    # config_path = './config/'+model+'.yaml'
    config_path = './config/' + model + '.yaml'
    with open(config_path, 'r') as f:
        dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    for key in dict:
        config[key] = dict[key]
    return config



if __name__ == "__main__":

    dataset = 'elec'
    model = 'SRGNN'

    train_method = 'random_ce'
    if 'Time' in model and 'ce' in train_method:
        train_method += '_time'

    gpu_id = 1
    sample_round = 3

    K = 10

    epochs = 300
    train_batch_size = 512


    # if 'nce' in train_method:
    #     sample_method = 'mn'
    # else:
    #     sample_method = 'random'


    config = load_config()
    # print(config['gpu_id'])
    cur_time,log_name,logger = load_logger(config)
    config['log_name'] = log_name

    max_len = config['max_len']


    prepare_data_model = PrepareData(config,logger)
    prepare_data_model.get_statistics()
    num_items = prepare_data_model.get_item_num()
    config['num_items'] = num_items
 






    train_loader, test_loader, dev_loader = load_data(prepare_data_model,config)

    founded_best_hit_10 = 0
    founded_best_ndcg_10 = 0

    founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5, \
    founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10, \
    founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20,founded_best_test_mae =\
    0,0,0,0,0,0,0,0,0,0

    # TODO 是不是可以从这里循环调参数
    best_config = config.copy()

    config_lst = load_hyper_param(config.copy(),model,dataset)
    hyper_count = len(config_lst)
    logger.info('[hyper parameter count]: %d' % (hyper_count))
    hyper_number = 0

    best_model_dict = None
    error_count = 0



    for config in config_lst:

        try:


            seed_torch()

            hyper_number += 1


            logger.info(' start training, running parameters:')
            logger.info(config)



            runner = TrainRunnerTime(
                config,
                train_loader,
                test_loader,
                dev_loader,
                logger
            )

            best_test_hit_5, best_test_ndcg_5, best_test_mrr_5, \
            best_test_hit_10, best_test_ndcg_10, best_test_mrr_10, \
            best_test_hit_20, best_test_ndcg_20, best_test_mrr_20, \
            best_test_mae, best_hr_10, best_ndcg_10=\
            runner.train(config['epochs'])



            if best_hr_10 > founded_best_hit_10 and best_ndcg_10> founded_best_ndcg_10:
                founded_best_hit_10 = best_hr_10
                founded_best_ndcg_10 = best_ndcg_10
                best_config = config.copy()
                best_model_dict = runner.get_best_model()


                founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5, \
                founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10, \
                founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20, founded_best_test_mae= \
                best_test_hit_5, best_test_ndcg_5, best_test_mrr_5,\
                best_test_hit_10, best_test_ndcg_10, best_test_mrr_10,\
                best_test_hit_20, best_test_ndcg_20, best_test_mrr_20, best_test_mae


                def save_load_best_model():


                    '''

                    cur_config['learning_rate'] = learning_rate
                                cur_config['n_layers'] = n_layers
                                cur_config['n_heads'] = n_heads
                                cur_config['hidden_dropout_prob'] = hidden_dropout_prob
                                cur_config['attn_dropout_prob'] = attn_dropout_prob
                    '''

                    model_save_path = 'data/model/' + model + '_' + dataset + '_' + str(cur_time) +'_' +config['train_method']+'_'+\
                                      str(best_config['learning_rate'])+'.pkl'
                                      # str(best_config['n_layers'])+'_'+\
                                      # str(best_config['n_heads']) + '_'+\
                                      # str(best_config['hidden_dropout_prob'])+'_'+\
                                      # str(best_config['attn_dropout_prob'])+'.pkl'
                    best_result_save_path = 'data/best_result/' + model + '_' + dataset + '_' + str(cur_time) + '_' + config['train_method']+'_'+\
                                      str(best_config['learning_rate'])+'.txt'
                                      # str(best_config['n_layers'])+'_'+\
                                      # str(best_config['n_heads']) + '_'+\
                                      # str(best_config['hidden_dropout_prob'])+'_'+\
                                      # str(best_config['attn_dropout_prob'])+'.txt'
                    best_config_save_path = 'data/model/' + model + '_' + dataset + '_' + str(cur_time) + '_' + config['train_method']+'_'+\
                                            str(best_config['learning_rate']) + '_config.json'
                                            # str(best_config['n_layers']) + '_' + \
                                            # str(best_config['n_heads']) + '_' + \
                                            # str(best_config['hidden_dropout_prob']) + '_' + \
                                            # str(best_config['attn_dropout_prob']) + '_config.json'
                    # TODO 暂时不保存
                    with open(best_config_save_path,'w') as w:
                        json_str = json.dumps(best_config,indent=4)
                        w.write(json_str)
                    # print('[best model path]' + model_save_path)
                    logger.info('[best model path]' + model_save_path)
                    # best_model = runner.model

                    th.save(best_model_dict, model_save_path)


                save_load_best_model()



            logger.info('finished')
            logger.info('[current config]')
            logger.info(config)
            logger.info('[best config]')
            logger.info(best_config)
            logger.info(
                '[score]: founded best [hit@10: %.5f, ndcg@10: %.5f], current [hit@10: %.5f, ndcg@10: %.5f]'
                % (founded_best_hit_10, founded_best_ndcg_10, best_hr_10, best_ndcg_10))
            logger.info('[hyper number]: %d/%d'%(hyper_number,hyper_count))

            logger.info('<founded best test> hit@5: %.5f, ndcg@5: %.5f, mrr@5: %.5f,'
                        'hit@10: %.5f, ndcg@10: %.5f, mrr@10: %.5f,'
                        'hit@20: %.5f, ndcg@20: %.5f, mrr@20: %.5f, mae: %.5f'
                        % (founded_best_test_hit_5, founded_best_test_ndcg_5, founded_best_test_mrr_5,
                            founded_best_test_hit_10, founded_best_test_ndcg_10, founded_best_test_mrr_10,
                            founded_best_test_hit_20, founded_best_test_ndcg_20, founded_best_test_mrr_20,
                           founded_best_test_mae))
            logger.info('=================finished current search======================')
        except Exception as e:
            # print(e)
            m = traceback.format_exc()
            logger.info(m)
            error_count+=1
            logger.info('error configuration count: %d'%error_count)


