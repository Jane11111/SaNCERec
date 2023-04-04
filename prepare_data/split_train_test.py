# -*- coding: utf-8 -*-


import random
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing
import os
import copy
import random
import pickle
import time

np.random.seed(1234)


class prepare_data_base():


    def __init__(self, data_name,origin_data,length_of_user_history):


        self.length = []
        self.type =  data_name
        self.origin_data = origin_data
        self.length_of_user_history = length_of_user_history

        self.data_type_error = 0
        self.data_too_short = 0

        # make origin data dir
        self.dataset_path = '/data/zxl/MTAM-t2/data/training_testing_data/' + self.type + "_time_item_based_unidirection"

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        self.dataset_class_pkl = os.path.join(self.dataset_path, 'parameters_loo.pkl')
        self.dataset_class_train = os.path.join(self.dataset_path, 'train_data_time_loo.pt')
        self.dataset_class_test = os.path.join(self.dataset_path, 'test_data_time_loo.pt')
        self.dataset_class_dev = os.path.join(self.dataset_path, 'dev_data_time_loo.pt')


        self.origin_data = origin_data
        # Init index for items, users and categories
        self.map_process()


    # give the index of item and category
    def map_process(self):
        """
        Map origin_data to one-hot-coding except time.

        """
        item_le = preprocessing.LabelEncoder()
        user_le = preprocessing.LabelEncoder()
        cat_le = preprocessing.LabelEncoder()

        # get item id list
        item_id = item_le.fit_transform(self.origin_data["item_id"].tolist())
        self.item_count = len(set(item_id))

        # get user id list
        user_id = user_le.fit_transform(self.origin_data["user_id"].tolist())
        self.user_count = len(set(user_id))

        # get category id list
        cat_id = cat_le.fit_transform(self.origin_data["cat_id"].tolist())
        self.category_count = len(set(cat_id))

        self.item_category_dic = {}
        for i in range(0, len(item_id)):
            self.item_category_dic[item_id[i]] = cat_id[i]

        print("item Count :" + str(len(item_le.classes_)))
        print("user count is " + str(len(user_le.classes_)))
        print("category count is " + str(len(cat_le.classes_)))

        # _key:key的列表，_map:key的列表加编号
        self.origin_data['item_id'] = item_id
        self.origin_data['user_id'] = user_id
        self.origin_data['cat_id'] = cat_id

        # 根据reviewerID、unixReviewTime编号进行排序（sort_values：排序函数）
        self.origin_data = self.origin_data.sort_values(['user_id', 'time_stamp'])

        # 重新建立索引
        self.origin_data = self.origin_data.reset_index(drop=True)
        return self.user_count, self.item_count


    def get_train_test(self):


        self.data_set = []
        self.train_set = []
        self.test_set = []
        self.dev_set = []

        self.now_count = 0

        # data_handle_process为各子类都使用的函数
        self.origin_data.groupby(["user_id"]).filter(lambda x: self.data_handle_process(x))
        # self.format_train_test()

        random.shuffle(self.train_set)
        random.shuffle(self.test_set)
        random.shuffle(self.dev_set)

        print('Size of training set is ' + str(len(self.train_set)))
        print('Size of testing set is ' + str(len(self.test_set)))
        print('Size of dev set is ' + str(len(self.dev_set)))
        print('Data type error size  is ' + str(self.data_type_error))
        print('Data too short size is ' + str(self.data_too_short))


        # train text 和 test text 使用文本
        self.save(self.train_set, self.dataset_class_train)
        self.save(self.test_set, self.dataset_class_test)
        self.save(self.dev_set, self.dataset_class_dev)

        return self.train_set, self.test_set, self.dev_set

    def data_handle_process_base(self, x):
        behavior_seq = copy.deepcopy(x)

        behavior_seq = behavior_seq.sort_values(by=['time_stamp'], na_position='first')
        behavior_seq = behavior_seq.reset_index(drop=True)
        columns_value = behavior_seq.columns.values.tolist()
        if "user_id" not in columns_value:
            self.data_type_error = self.data_type_error + 1
            return

        self.now_count = self.now_count + 1
        # test
        behavior_seq = behavior_seq.reset_index(drop=True)
        return behavior_seq

    def proc_pos_emb(self, time_stamp_seq):
        interval_list = [i for i in range(0, len(time_stamp_seq))]
        return interval_list


    def pro_time_method(self, time_stamp_seq, mask_time):
        timelast_list = [time_stamp_seq[i+1]-time_stamp_seq[i] for i in range(0,len(time_stamp_seq)-1,1)]
        timelast_list.insert(0,0)
        timenow_list = [mask_time-time_stamp_seq[i] for i in range(0,len(time_stamp_seq),1)]

        return timelast_list,timenow_list

    def mask_process_unidirectional(self, index,length,user_seq,item_seq,category_seq,time_stamp_seq,
                                    lengeth_limit=50):

        # only sample the data before the label
        temp_index = index

        # 如果长度超标，进行截取
        if temp_index - lengeth_limit + 1 > 0:
            start = temp_index - lengeth_limit + 1
        else:
            start = 0

        user_seq_temp = [user_seq[i] for i in range(start, length) if i < temp_index]
        item_seq_temp = [item_seq[i] for i in range(start, length) if i < temp_index]
        category_seq_temp = [category_seq[i] for i in range(start, length) if i < temp_index]
        time_stamp_seq_temp = [time_stamp_seq[i] for i in range(start, length) if i < temp_index]

        factor_list = [category_seq_temp, time_stamp_seq_temp]

        return user_seq_temp[0], item_seq_temp, factor_list



    def convert_time(self,cur_time):
        cur_time = time.localtime(cur_time)
        year = cur_time.tm_year
        month = cur_time.tm_mon
        day = cur_time.tm_mday
        dayofweek = cur_time.tm_wday
        hour = cur_time.tm_hour
        return year,month,day,dayofweek,hour

    # def process_time(self,time_seq,target_time):
    #
    #
    #
    #     target_year,target_month,target_day,target_dayofweek,target_hour = self.convert_time(target_time)
    #     #
    #     # year_lst = []
    #     # month_lst = []
    #     # day_lst = []
    #     # dayofweek_lst = []
    #     # hour_lst = []
    #     #
    #     # for cur_time in time_seq:
    #     #     year,month,day,dayofweek,hour = self.convert_time(cur_time)
    #     #     year_lst.append(year)
    #     #     month_lst.append(month)
    #     #     day_lst.append(day)
    #     #     dayofweek_lst.append(dayofweek)
    #     #     hour_lst.append(hour_lst)
    #
    #     return [year_lst,month_lst,day_lst,dayofweek_lst,hour_lst],[target_year,target_month,target_day,target_dayofweek,target_hour]




    def data_handle_process(self, x):

        # 8份作为train，1份dev， 1份test

        random_num = np.random.rand()
        train_user = False
        test_user = False
        dev_user = False
        # if random_num <= 0.8:
        #     train_user = True
        # elif random_num <=0.9:
        #     test_user = True
        # else:
        #     dev_user = True


        behavior_seq = self.data_handle_process_base(x)

        if behavior_seq.shape != x.shape:
            print('error!!!!!!!!!!')

        if behavior_seq is None:
            return


        user_seq = behavior_seq['user_id'].tolist()
        item_seq = behavior_seq["item_id"].tolist()
        category_seq = behavior_seq["cat_id"].tolist()  # TODO modified
        # category_seq = list(np.ones_like(item_seq))
        time_stamp_seq = behavior_seq["time_stamp"].tolist()
        length = behavior_seq.shape[0]

        mask_index_list = [i for i in range(1, behavior_seq.shape[0])]

        for index in mask_index_list:

            # 这里只取单项
            user_id, item_seq_temp, factor_list = \
                self.mask_process_unidirectional(index=index,
                                                 length=length,
                                                 user_seq=user_seq,
                                                 item_seq=item_seq,
                                                 category_seq=category_seq,
                                                 time_stamp_seq=time_stamp_seq,
                                                 lengeth_limit=self.length_of_user_history)

            target_id = item_seq[index]



            # history_dates,target_dates = self.process_time(factor_list[1],time_stamp_seq[index])

            time_seq_temp = [int(x/3600) for x in factor_list[1]]
            target_time = int(time_stamp_seq[index]/3600)



            time_interval_seq = [time_seq_temp[i+1] - time_seq_temp[i] for i in range(len(time_seq_temp)-1)]
            time_interval_seq.insert(0,0)



            lst = (user_id, item_seq_temp ,  target_id , len(item_seq_temp),time_seq_temp,time_interval_seq,target_time )

            # 以小时为准
            # if test_user:
            #     self.test_set.append(lst)
            # elif dev_user:
            #     self.dev_set.append(lst)
            # else:
            #     self.train_set.append(lst)
            if index == mask_index_list[-1]:
                self.test_set.append(lst)
            elif index == mask_index_list[-2]:
                self.dev_set.append(lst)
            else:
                self.train_set.append(lst)
    # 给出写入文件
    def save(self, data_list, file_path):
        fp = open(file_path, 'wb')
        # for i in data_list:
        #     fp.write(str(i) + '\n')
        fp.write(pickle.dumps(data_list))
        fp.close()


