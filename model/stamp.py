# -*- coding: utf-8 -*-





import torch
from torch import nn

from modules.abstract_recommender import SequentialRecommender
from modules.layers import TransformerEncoder
from torch.nn.init import xavier_uniform_, xavier_normal_


class STAMP(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(STAMP, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w0 = nn.Linear(self.embedding_size, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=True)
        self.mlp_a = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.mlp_b = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # # parameters initialization
        # self.apply(self._init_weights)
        print('............initializing................')


    # def _init_weights(self, module):
    #     if isinstance(module, nn.Embedding):
    #         xavier_normal_(module.weight)
    #     elif isinstance(module, nn.GRU):
    #         xavier_uniform_(self.gru_layers.weight_hh_l0)
    #         xavier_uniform_(self.gru_layers.weight_ih_l0)



    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        last_inputs = self.gather_indexes(item_seq_emb, item_seq_len - 1)
        org_memory = item_seq_emb
        ms = torch.div(torch.sum(org_memory, dim=1), item_seq_len.unsqueeze(1).float())
        alpha = self.count_alpha(org_memory, last_inputs, ms)
        vec = torch.matmul(alpha.unsqueeze(1), org_memory)
        ma = vec.squeeze(1) + ms
        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last_inputs))
        seq_output = hs * ht
        return seq_output

    def count_alpha(self, context, aspect, output):

        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        output_3dim = output.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha


    def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        return logits



    def calculate_logits_given_items(self,item_seq,item_seq_len,pos_neg_items, time_seq=None,time_interval_seq=None,target_time=None):
        """

        :param item_seq:
        :param item_seq_len:
        :param actual_noise_items: B, 1+K
        :return:
        """
        seq_output = self.forward(item_seq,item_seq_len) # B, emb_size

        actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size

        actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K

        return actual_noise_scores


