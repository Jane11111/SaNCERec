# -*- coding: utf-8 -*-




import torch
from torch import nn

from modules.abstract_recommender import SequentialRecommender
from modules.layers import TransformerEncoder
from torch.nn.init import xavier_uniform_, xavier_normal_


class GRU4Rec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        # self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        # self.dropout_prob = config['dropout_prob']

        self.device = config['device']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)

        # parameters initialization
        # self.apply(self._init_weights)
        # print('............initializing................')

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(self.gru_layers.weight_hh_l0)
            xavier_uniform_(self.gru_layers.weight_ih_l0)



    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        # item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb)
        # gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)



        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        seq_output = self.LayerNorm(seq_output)
        return seq_output

    def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        return logits

    def calculate_logits2(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        return logits,seq_output

    def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        return seq_output,logits


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




    def get_noise_score(self,item_seq,item_seq_len,target_id,K):
        """

        :param item_seq:
        :param item_seq_len:
        :param target_id: B*1
        :param K:
        :return: B*K
        """
        device = item_seq.device
        softmax_fun = nn.Softmax(dim = 1)
        logits = softmax_fun(self.calculate_logits(item_seq,item_seq_len) )
        ones = torch.ones( logits.shape  ).to(device)
        one_hot = ones.scatter_(1, target_id.reshape(-1,1) , 0)
        logits *= one_hot

        topk_score,topk = logits.topk(k=K)
        return topk_score,topk

    def get_target_score(self,item_seq,item_seq_len,target_id):
        seq_output = self.forward(item_seq, item_seq_len)  # B*emb_size
        target_emb = self.item_embedding(target_id).squeeze(1)  # B*emb_size
        target_score = torch.sum(seq_output * target_emb, axis=1)
        return target_score
