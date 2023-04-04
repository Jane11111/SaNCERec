# -*- coding: utf-8 -*-


import torch
from torch import nn
import numpy as np

from modules.abstract_recommender import SequentialRecommender
from modules.layers import TransformerEncoder


class CSASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(CSASRec, self).__init__(config, dataset)

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
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
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

        self.time_linear = nn.Linear(1,self.hidden_size)
        self.softplus_fun = nn.Softplus()
        self.eps = np.finfo(float).eps
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

        new_time_seq = torch.unsqueeze(time_seq, 2)
        target_interval = target_time.reshape(-1,1)-self.gather_indexes(new_time_seq, item_seq_len - 1) # B, 1
        interval_emb = self.time_linear(target_interval)

        seq_output = seq_output+interval_emb

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        logits = self.softplus_fun(logits)

        return logits

    def  get_total_intensities(self,seq_output,sampled_dt):
        """

        :param seq_output: B*emb_size
        :param sampled_dt: B*K
        :return:
        """
        new_sampled_dt = sampled_dt.unsqueeze(-1) # B,K,1
        sampled_dt_emb = self.time_linear(new_sampled_dt) # B,K,emb_size

        seq_time_emb = seq_output.unsqueeze(1)+sampled_dt_emb # B,K,emb_size

        item_emb = self.item_embedding.weight # item_num, emb_size
        total_item_emb = torch.sum(item_emb,dim = 0) # emb_size,

        total_intensity = self.softplus_fun(torch.sum(seq_time_emb*total_item_emb.unsqueeze(0).unsqueeze(0), dim = 2) )# B,K

        return total_intensity
    def get_intensity_give_time_type(self,time_lst,type_lst,seq_output):
        """

        :param time_lst: B * (1+K)
        :param type_lst: B * (1+K)
        :param seq_output: B * emb_size
        :return:
        """
        item_emb = self.item_embedding(type_lst)
        time_emb = self.time_linear(time_lst.unsqueeze(-1)) # B,(1+K),emb_size
        emb = item_emb + time_emb # B, (1+K),emb_size

        scores = torch.sum(emb*seq_output.unsqueeze(1),dim=2) # B,(1+K)
        scores = self.softplus_fun(scores)

        return scores

    def sample_noises(self,item_seq,item_seq_len,time_seq,time_interval_seq,target_time,sample_num_max,probs):
        """
        :param item_seq:
        :param item_seq_len:
        :param time_seq:
        :param time_interval_seq:
        :return:
        """
        device = item_seq.device
        seq_output = self.forward(item_seq, item_seq_len)

        new_time_seq = torch.unsqueeze(time_seq, 2)
        target_interval = target_time.reshape(-1,1)-self.gather_indexes(new_time_seq, item_seq_len - 1)  # B, 1

        M1 = sample_num_max

        sample_rate = 1/(target_interval+self.eps)
        batch_size,D = seq_output.size()

        Exp_numbers = torch.empty(
            size = [batch_size,sample_num_max],dtype= torch.float32
        ).to(device)
        Unif_numbers = torch.empty(
            size = [batch_size,sample_num_max],dtype= torch.float32
        ).to(device)
        Exp_numbers.exponential_(1.0)
        sampled_dt = Exp_numbers /(sample_rate * M1)
        sampled_dt = sampled_dt.cumsum(dim=-1) #B,K
        sampled_type = torch.randint(0,self.n_items ,size=sampled_dt.shape ).to(device) # B,K
        # sampled_type = torch.ones(sampled_dt.shape,dtype=torch.long).to(device)

        total_noise_intensity = self.get_total_intensities(seq_output,sampled_dt) # B,K
        to_collect = sampled_dt<target_interval
        mask_noise, accept_prob = self.thining(Unif_numbers,sample_rate,total_noise_intensity, to_collect)




        return sampled_dt,sampled_type,mask_noise,accept_prob

    def thining(self,
                Unif_numbers,sample_rate,total_noise_intensity, to_collect ):
        """
        对于采样得到的sampled_dt，拒绝一部分value
        :param Unif_numbers: B , K
        :param sample_rate: B, 1
        :param total_noise_intensity: B,K
        :param sampled_dt: B,K
        :param to_collect: B,K
        :param sample_num_max:
        :return: noise_mask B , K
        """
        accept_prob = total_noise_intensity/(sample_rate+self.eps)
        accept_prob[accept_prob>1] = 1
        Unif_numbers.uniform_(0.0,1.0)

        threshold = 0.01

        id1 = accept_prob<threshold
        id2 = Unif_numbers < accept_prob
        to_collect[id1*(~id2)] = 0

        accept_prob[id1 * id2] = 1.0

        mask_noise = to_collect.float()

        return mask_noise,accept_prob










