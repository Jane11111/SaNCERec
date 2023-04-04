# -*- coding: utf-8 -*-


import torch
from torch import nn

from modules.abstract_recommender import SequentialRecommender
from modules.layers import TransformerEncoder


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = float(config['layer_norm_eps'])

        self.initializer_range = config['initializer_range']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.device = config['device']


        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        # print(item_seq.size(1))
        # print(item_seq.device)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

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

    # def get_nce_loss(self,item_seq,item_seq_len,noise_score,target_id):
    #
    #
    #     seq_output = self.forward(item_seq,item_seq_len)  # B*emb_size
    #     target_emb = self.item_embedding(target_id).squeeze(1) # B*emb_size
    #     target_score = torch.sum(seq_output*target_emb,axis = 1)
    #
    #     loss = target_score/(torch.sum(noise_score,axis=1)+target_score)
    #     return loss
