# -*- coding: utf-8 -*-




import torch
from torch import nn

from modules.abstract_recommender import SequentialRecommender
from modules.layers import TransformerEncoder
from torch.nn.init import xavier_uniform_, xavier_normal_


class ModifiedGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModifiedGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size

        self.reset_linear = nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)
        self.update_linear = nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)
        self.candidate_linear = nn.Linear(self.embedding_size + self.hidden_size, self.hidden_size)

        self.layer1 = nn.Linear(self.embedding_size*2, self.hidden_size)
        self.layer2 = nn.Linear(1, self.hidden_size)

        self._time_kernel_w1 = nn.Parameter(torch.ones((self.embedding_size)))
        self._time_kernel_b1 = nn.Parameter(torch.ones((self.embedding_size)))
        self._time_history_w1 = nn.Parameter(torch.ones((self.embedding_size)))

        self._time_w1 = nn.Parameter(torch.ones((self.embedding_size)))
        self._time_b1 = nn.Parameter(torch.ones((self.embedding_size)))

        self._time_kernel_w2 = nn.Parameter(torch.ones((self.embedding_size)))
        self._time_w12 = nn.Parameter(torch.ones((self.embedding_size)))

        self._time_b12 = nn.Parameter(torch.ones((self.embedding_size)))

        self.new_r_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.new_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.relu_activation = torch.nn.ReLU()

    def forward(self, inputs, state):
        time_interval = inputs[:,-1].reshape(-1,1) # B,1
        inputs = inputs[:, :self.embedding_size]

        gate_inputs = torch.cat((inputs, state), 1)

        r = torch.sigmoid(self.reset_linear(gate_inputs))
        u = torch.sigmoid(self.update_linear(gate_inputs))

        r_state = r * state

        candidate = self.candidate_linear(torch.cat((inputs, r_state), 1))
        c = torch.tanh(candidate)

        # semantic_emb = self.relu_activation(self.layer1(torch.cat((inputs,state),1)))
        # time_emb = self.relu_activation(self.layer2(time_interval))
        # new_gate = torch.sigmoid(self.new_gate(torch.cat((semantic_emb, time_emb), 1)))

        time_last_weight = self.relu_activation(inputs*self._time_kernel_w1 + state*self._time_history_w1+self._time_kernel_b1)
        # time_last_weight = self.relu_activation(
        #     inputs * self._time_kernel_w1 +  self._time_kernel_b1)

        time_last_score = self.relu_activation(self._time_w1*time_interval + self._time_b1)
        time_last_state = torch.sigmoid(self._time_kernel_w2*time_last_weight + self._time_w12*time_last_score+self._time_b12)
        new_gate = time_last_state


        new_h = u * state + (1 - u) * c * new_gate

        return new_h, new_h

class ModifiedGRU(nn.Module):

    def __init__(self,input_size, hidden_size):

        super(ModifiedGRU,self).__init__()

        self.cell = ModifiedGRUCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self,x,h=None):

        batch_size, length = x.size(0), x.size(1)

        outputs = torch.empty(batch_size,length,self.hidden_size).to(x.device)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size,device = x.device )
        state = h
        for l in range(length):
            # if l == 49:
            #     print('i am here')
            output, state = self.cell(x[:,l,:],state)

            # outputs.append(output)
            outputs[:,l,:] = output

        return outputs, state

# class TimeAutoRegrGRU4Rec(SequentialRecommender):
#     r"""
#     SASRec is the first sequential recommender based on self-attentive mechanism.
#     NOTE:
#         In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
#         by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
#         using Fully Connected Layer to implement the PFFN.
#     """
#
#     def __init__(self, config, dataset):
#         super(TimeAutoRegrGRU4Rec, self).__init__(config, dataset)
#
#         # load parameters info
#         self.embedding_size = config['embedding_size']
#         self.hidden_size = config['hidden_size']
#         # self.loss_type = config['loss_type']
#         # self.num_layers = config['num_layers']
#         # self.dropout_prob = config['dropout_prob']
#
#         # define layers and loss
#         self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
#         # self.emb_dropout = nn.Dropout(self.dropout_prob)
#         self.gru_layers =  ModifiedGRU(
#             input_size=self.embedding_size+1,
#             hidden_size=self.hidden_size,
#
#         )
#         # self.gru_layers = nn.GRU(
#         #     input_size=self.embedding_size  ,
#         #     hidden_size=self.hidden_size,
#         #
#         # )
#         self.time_embedding = nn.Embedding(31, self.hidden_size)
#
#         self.dense = nn.Linear(self.hidden_size, self.embedding_size)
#         self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
#         # parameters initialization
#         # self.apply(self._init_weights)
#         print('............initializing................')
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Embedding):
#             xavier_normal_(module.weight)
#         elif isinstance(module, nn.GRU):
#             xavier_uniform_(self.gru_layers.weight_hh_l0)
#             xavier_uniform_(self.gru_layers.weight_ih_l0)
#
#
#
#     def forward(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#         item_seq_emb = self.item_embedding(item_seq)
#         # item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
#
#         # time_interval_seq = time_interval_seq / 24 #以天为单位
#         # time_interval_seq[time_interval_seq > 30] = 30  # 把最大值限制为30
#         # diff = abs(time_interval_seq)
#         # time_embedding = self.time_embedding(diff.long())
#         # gru_inputs = item_seq_emb + time_embedding
#
#         gru_inputs = torch.cat((item_seq_emb,time_interval_seq.unsqueeze(-1)),2) # B,max_len, num_units+1
#
#         gru_output, _ = self.gru_layers(gru_inputs)
#
#         # gru_output = self.dense(gru_output)
#         # the embedding of the predicted item, shape of (batch_size, embedding_size)
#         seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
#
#         seq_output = self.LayerNorm(seq_output)
#         return seq_output
#
#     def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return logits
#
#     def calculate_logits2(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return logits,seq_output
#
#     def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return seq_output,logits
#
#
#     def calculate_logits_given_items(self,item_seq,item_seq_len,pos_neg_items, time_seq=None,time_interval_seq=None,target_time=None):
#         """
#
#         :param item_seq:
#         :param item_seq_len:
#         :param actual_noise_items: B, 1+K
#         :return:
#         """
#         seq_output = self.forward(item_seq,item_seq_len, time_seq,time_interval_seq,target_time) # B, emb_size
#
#         actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size
#
#         actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K
#
#         return actual_noise_scores



class TimePredGRU4Rec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(TimePredGRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        # self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        # self.dropout_prob = config['dropout_prob']

        self.device = config['device']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items+1, self.embedding_size)
        # self.emb_dropout = nn.Dropout(self.dropout_prob)
        # self.gru_layers = nn.GRU(
        #     input_size=self.embedding_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        #     bias=False,
        #     batch_first=True,
        # )
        self.gru_layers = ModifiedGRU(
            input_size=self.embedding_size + 1,
            hidden_size=self.hidden_size,

        )

        self.output_weight = nn.Parameter(torch.ones(self.hidden_size, ) * 0.5)

        # self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)

        self.time_linear = nn.Linear(self.hidden_size,1)
        # parameters initialization
        # self.apply(self._init_weights)
        print('............initializing................')

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)



    def get_avg_intervals(self,time_seq,item_seq_len):

        time_intervals = time_seq[:, 1:] - time_seq[:, :-1]
        zero_intervals = torch.zeros(item_seq_len.shape).reshape(-1, 1).to(time_seq.device)
        time_intervals = torch.cat([zero_intervals, time_intervals], 1)
        time_intervals[time_intervals < 0] = 0  # batch_size, max_seq_len
        avg_time_intervals = torch.sum(time_intervals, 1) / item_seq_len.reshape(-1, )  # batch_size, 1
        avg_time_intervals = avg_time_intervals.reshape(-1,1)


        return avg_time_intervals





    def forward(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        mask_index = torch.ones(target_time.shape).long().to(item_seq.device)*self.n_items
        new_item_seq = item_seq.scatter(1, item_seq_len.reshape(-1, 1), mask_index.reshape(-1, 1))
        item_seq_emb = self.item_embedding(new_item_seq)

        avg_time_intervals = self.get_avg_intervals(time_seq, item_seq_len)
        new_time_interval_seq = time_interval_seq.scatter(1,item_seq_len.reshape(-1,1),avg_time_intervals)

        item_seq_len = item_seq_len + 1
        # item_seq_len += 1


        gru_inputs = torch.cat((item_seq_emb,new_time_interval_seq.unsqueeze(-1)),2)

        gru_output, _ = self.gru_layers(gru_inputs)
        # gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)

        output1 = self.gather_indexes(gru_output, item_seq_len - 1) # 最后两个一起预测
        output2 = self.gather_indexes(gru_output, item_seq_len - 2)

        output = output1 * self.output_weight + output2 * (1 - self.output_weight)

        time_intervals = self.time_linear(output)

        return output,time_intervals
    def lambda_w_time(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        mask_index = torch.ones(target_time.shape).long().to(item_seq.device)*self.n_items
        new_item_seq = item_seq.scatter(1, item_seq_len.reshape(-1, 1), mask_index.reshape(-1, 1))
        item_seq_emb = self.item_embedding(new_item_seq)

        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        target_time_interval = target_time.reshape(-1, 1) - last_time

        # target_time时刻的lambda
        new_time_interval_seq = time_interval_seq.scatter(1,item_seq_len.reshape(-1,1),target_time_interval )

        # # TODO modified
        # new_time_interval_seq2 = time_interval_seq.scatter(1, item_seq_len.reshape(-1, 1), target_time_interval / 10)
        # # TODO end

        item_seq_len = item_seq_len + 1

        gru_inputs = torch.cat((item_seq_emb,new_time_interval_seq.unsqueeze(-1)),2)

        gru_output, _ = self.gru_layers(gru_inputs)
        # gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)

        # print(item_seq_len)

        output1 = self.gather_indexes(gru_output, item_seq_len - 1) # 最后两个一起预测
        output2 = self.gather_indexes(gru_output, item_seq_len - 2)

        # # TODO modified
        #
        # gru_inputs = torch.cat((item_seq_emb, new_time_interval_seq2.unsqueeze(-1)), 2)
        #
        # gru_output2, _ = self.gru_layers(gru_inputs)
        # # gru_output = self.dense(gru_output)
        # # the embedding of the predicted item, shape of (batch_size, embedding_size)
        #
        # output3 = self.gather_indexes(gru_output2, item_seq_len - 1)  # 最后两个一起预测
        # output4 = self.gather_indexes(gru_output2, item_seq_len - 2)
        #
        # # TODO end

        output = output1 * self.output_weight + output2 * (1 - self.output_weight)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(output, test_item_emb.transpose(0, 1))
        logits = logits[:, :self.n_items]

        logits[logits<0] = 0

        lambda_val = torch.sum(logits,1)

        return lambda_val


    def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        seq_output,_ = self.forward(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        logits = logits[:, :self.n_items]
        return logits

    def calculate_logits_time(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        seq_output,predict_intervals = self.forward(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        logits = logits[:, :self.n_items]
        return logits, predict_intervals

    def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        seq_output,_ = self.forward(item_seq, item_seq_len, time_seq,time_interval_seq,target_time)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        logits = logits[:, :self.n_items]
        return seq_output,logits


    def calculate_logits_given_items(self,item_seq,item_seq_len,pos_neg_items, time_seq=None,time_interval_seq=None,target_time=None):
        """

        :param item_seq:
        :param item_seq_len:
        :param actual_noise_items: B, 1+K
        :return:
        """
        seq_output,_ = self.forward(item_seq,item_seq_len, time_seq,time_interval_seq,target_time) # B, emb_size

        actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size

        actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K

        return actual_noise_scores


    def calculate_logits_time_given_items(self,item_seq,item_seq_len,pos_neg_items, time_seq,time_interval_seq,target_time):
        """

        :param item_seq:
        :param item_seq_len:
        :param actual_noise_items: B, 1+K
        :return:
        """
        seq_output,predict_intervals = self.forward(item_seq,item_seq_len, time_seq,time_interval_seq,target_time) # B, emb_size

        actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size

        actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K

        return actual_noise_scores, predict_intervals


    # def get_noise_score(self,item_seq,item_seq_len,target_id,K):
    #     """
    #
    #     :param item_seq:
    #     :param item_seq_len:
    #     :param target_id: B*1
    #     :param K:
    #     :return: B*K
    #     """
    #     device = item_seq.device
    #     softmax_fun = nn.Softmax(dim = 1)
    #     logits = softmax_fun(self.calculate_logits(item_seq,item_seq_len) )
    #     ones = torch.ones( logits.shape  ).to(device)
    #     one_hot = ones.scatter_(1, target_id.reshape(-1,1) , 0)
    #     logits *= one_hot
    #
    #     topk_score,topk = logits.topk(k=K)
    #     return topk_score,topk
    #
    # def get_target_score(self,item_seq,item_seq_len,target_id):
    #     seq_output = self.forward(item_seq, item_seq_len)  # B*emb_size
    #     target_emb = self.item_embedding(target_id).squeeze(1)  # B*emb_size
    #     target_score = torch.sum(seq_output * target_emb, axis=1)
    #     return target_score





