# -*- coding: utf-8 -*-




import torch
from torch import nn

from modules.abstract_recommender import SequentialRecommender
from modules.layers import TimeAwareTransformerEncoder,TransformerEncoder
from modules.layers import TimeAwareTransformerEncoder2
from modules.layers import TimePredTransformerEncoder

# class TimeAwareSASRec(SequentialRecommender):
#     r"""
#     SASRec is the first sequential recommender based on self-attentive mechanism.
#     NOTE:
#         In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
#         by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
#         using Fully Connected Layer to implement the PFFN.
#     """
#
#     def __init__(self, config, dataset):
#         super(TimeAwareSASRec, self).__init__(config, dataset)
#
#         # load parameters info
#         self.n_layers = config['n_layers']
#         self.n_heads = config['n_heads']
#         self.hidden_size = config['hidden_size']  # same as embedding_size
#         self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
#         self.hidden_dropout_prob = config['hidden_dropout_prob']
#         self.attn_dropout_prob = config['attn_dropout_prob']
#         self.hidden_act = config['hidden_act']
#         self.layer_norm_eps = float(config['layer_norm_eps'])
#
#         self.initializer_range = config['initializer_range']
#
#         # define layers and loss
#         self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)
#         self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
#         self.trm_encoder = TimeAwareTransformerEncoder(
#             n_layers=self.n_layers,
#             n_heads=self.n_heads,
#             max_seq_len= self.max_seq_length,
#             hidden_size=self.hidden_size,
#             inner_size=self.inner_size,
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             attn_dropout_prob=self.attn_dropout_prob,
#             hidden_act=self.hidden_act,
#             layer_norm_eps=self.layer_norm_eps
#         )
#
#         self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
#         self.dropout = nn.Dropout(self.hidden_dropout_prob)
#
#         self.device = config['device']
#
#
#         # parameters initialization
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.initializer_range)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#
#     def get_attention_mask(self, item_seq):
#         """Generate left-to-right uni-directional attention mask for multi-head attention."""
#         attention_mask = (item_seq > 0).long()
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#         # mask for left-to-right unidirectional
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#         subsequent_mask = subsequent_mask.long().to(item_seq.device)
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         return extended_attention_mask
#
#     def forward(self, item_seq, item_seq_len,time_seq):
#         # print(item_seq.size(1))
#         # print(item_seq.device)
#         position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
#         position_embedding = self.position_embedding(position_ids)
#
#         item_emb = self.item_embedding(item_seq)
#         input_emb = item_emb + position_embedding
#         input_emb = self.LayerNorm(input_emb)
#         input_emb = self.dropout(input_emb)
#
#         extended_attention_mask = self.get_attention_mask(item_seq)
#
#         trm_output = self.trm_encoder(input_emb,time_seq, extended_attention_mask, output_all_encoded_layers=True)
#         output = trm_output[-1]
#         output = self.gather_indexes(output, item_seq_len - 1)
#         return output  # [B H]
#
#     def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return logits
#
#     def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq)
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
#         seq_output = self.forward(item_seq,item_seq_len,time_seq) # B, emb_size
#
#         actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size
#
#         actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K
#
#         return actual_noise_scores
class TimePredSASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.
    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(TimePredSASRec, self).__init__(config, dataset)

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
        self.item_embedding = nn.Embedding(self.n_items+1, self.hidden_size)
        self.time_embedding = nn.Embedding(13,self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TimePredTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            max_seq_len= self.max_seq_length,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.time_linear = nn.Linear(self.hidden_size+self.max_seq_length,1)

        self.output_weight = nn.Parameter(torch.ones(self.hidden_size,)*0.5)

        self.relu_fun = nn.ReLU()

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


    def get_attention_mask (self,  item_seq_len ):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        batch_size = item_seq_len.shape[0]
        seq_range = torch.arange(0, self.max_seq_length).long().to(item_seq_len.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, self.max_seq_length)
        # print(seq_range_expand)

        seq_len_expand = item_seq_len.reshape(-1, 1).repeat(1, self.max_seq_length)
        # print(seq_len_expand)

        attention_mask = (seq_range_expand<seq_len_expand).long()

        # attention_mask = (new_item_seq > 0).long() # batch_size, max_len


        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64 512，1，1，50
        # mask for the whole sequence
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        # subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        # subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        # subsequent_mask = subsequent_mask.long().to(item_seq.device) # 1，1，50，50
        subsequent_mask = torch.ones(attn_shape).unsqueeze(0).to(item_seq_len.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask




    def get_avg_intervals(self,time_seq,item_seq_len):

        time_intervals = time_seq[:, 1:] - time_seq[:, :-1]
        zero_intervals = torch.zeros(item_seq_len.shape).reshape(-1, 1).to(time_seq.device)
        time_intervals = torch.cat([zero_intervals, time_intervals], 1)
        time_intervals[time_intervals < 0] = 0  # batch_size, max_seq_len
        avg_time_intervals = torch.sum(time_intervals, 1) / item_seq_len.reshape(-1, )  # batch_size, 1
        avg_time_intervals = avg_time_intervals.reshape(-1,1)


        return avg_time_intervals


    def forward(self, item_seq, item_seq_len,time_seq,target_time):

        """
        :param item_seq:
        :param item_seq_len:
        :param time_seq: batch_size,max_seq_len
        :param target_time: batch_size,
        :return:
        """


        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        mask_index = torch.ones(target_time.shape).long().to(item_seq.device)*self.n_items
        new_item_seq = item_seq.scatter(1, item_seq_len.reshape(-1, 1), mask_index.reshape(-1, 1))

        item_emb = self.item_embedding(new_item_seq)
        input_emb = item_emb + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)


        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        avg_time_intervals = self.get_avg_intervals(time_seq, item_seq_len)
        coarse_target_time = last_time + avg_time_intervals
        new_time_seq = time_seq.scatter(1,item_seq_len.reshape(-1,1),coarse_target_time.reshape(-1,1))
        # new_time_seq = time_seq


        item_seq_len = item_seq_len+1

        extended_attention_mask = self.get_attention_mask(item_seq_len) # 多用一个mask位

        trm_output,intervals = self.trm_encoder(input_emb,new_time_seq, item_seq_len,extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output1 = self.gather_indexes(output, item_seq_len-1)
        output2 = self.gather_indexes(output, item_seq_len - 2)

        output = output1*self.output_weight + output2*(1-self.output_weight)

        intervals = intervals[-1]

        return output , intervals  # [B H]



    def lambda_w_time(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        mask_index = torch.ones(target_time.shape).long().to(item_seq.device) * self.n_items
        new_item_seq = item_seq.scatter(1, item_seq_len.reshape(-1, 1), mask_index.reshape(-1, 1))

        item_emb = self.item_embedding(new_item_seq)
        input_emb = item_emb + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)


        new_time_seq = time_seq.scatter(1, item_seq_len.reshape(-1, 1), target_time.reshape(-1, 1))


        item_seq_len = item_seq_len + 1


        extended_attention_mask = self.get_attention_mask(item_seq_len)  # 多用一个mask位

        trm_output, intervals = self.trm_encoder(input_emb, new_time_seq, item_seq_len, extended_attention_mask,
                                                 output_all_encoded_layers=True)
        output = trm_output[-1]
        output1 = self.gather_indexes(output, item_seq_len - 1)
        output2 = self.gather_indexes(output, item_seq_len - 2)

        output = output1 * self.output_weight + output2 * (1 - self.output_weight)



        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(output, test_item_emb.transpose(0, 1))
        logits = logits[:, :self.n_items]

        logits[logits<0] = 0

        lambda_val = torch.sum(logits,1)

        return lambda_val

    def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):



        seq_output ,_= self.forward(item_seq, item_seq_len,time_seq,target_time)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        logits = logits[:,:self.n_items]
        return logits

    def calculate_logits_time(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):



        seq_output,predict_intervals = self.forward(item_seq, item_seq_len,time_seq,target_time)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

        # avg_intervals = self.get_avg_intervals(time_seq,item_seq_len)
        # predict_intervals = self.time_linear(torch.cat([seq_output,time_interval_seq ],1) )

        logits = logits[:,:self.n_items]
        return logits, predict_intervals

    def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):

        seq_output = self.forward(item_seq, item_seq_len,time_seq,target_time)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        logits = logits[:,:self.n_items]
        return seq_output,logits


    def calculate_logits_time_given_items(self,item_seq,item_seq_len,pos_neg_items, time_seq=None,time_interval_seq=None,target_time=None):
        """

        :param item_seq:
        :param item_seq_len:
        :param actual_noise_items: B, 1+K
        :return:
        """
        seq_output,predict_intervals = self.forward(item_seq,item_seq_len,time_seq,target_time) # B, emb_size

        actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size

        actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K

        return actual_noise_scores,predict_intervals


#
# class TimeAutoRegrSASRec(SequentialRecommender):
#     r"""
#     SASRec is the first sequential recommender based on self-attentive mechanism.
#     NOTE:
#         In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
#         by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
#         using Fully Connected Layer to implement the PFFN.
#     """
#
#     def __init__(self, config, dataset):
#         super(TimeAutoRegrSASRec, self).__init__(config, dataset)
#
#         # load parameters info
#         self.n_layers = config['n_layers']
#         self.n_heads = config['n_heads']
#         self.hidden_size = config['hidden_size']  # same as embedding_size
#         self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
#         self.hidden_dropout_prob = config['hidden_dropout_prob']
#         self.attn_dropout_prob = config['attn_dropout_prob']
#         self.hidden_act = config['hidden_act']
#         self.layer_norm_eps = float(config['layer_norm_eps'])
#
#         self.initializer_range = config['initializer_range']
#
#         # define layers and loss
#         self.item_embedding = nn.Embedding(self.n_items+1, self.hidden_size)
#         self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
#         self.trm_encoder = TimeAwareTransformerEncoder(
#             n_layers=self.n_layers,
#             n_heads=self.n_heads,
#             max_seq_len= self.max_seq_length,
#             hidden_size=self.hidden_size,
#             inner_size=self.inner_size,
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             attn_dropout_prob=self.attn_dropout_prob,
#             hidden_act=self.hidden_act,
#             layer_norm_eps=self.layer_norm_eps
#         )
#
#         self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
#         self.dropout = nn.Dropout(self.hidden_dropout_prob)
#
#         self.time_linear = nn.Linear(self.hidden_size,1)
#
#         self.relu_fun = nn.ReLU()
#
#         self.device = config['device']
#
#
#         # parameters initialization
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.initializer_range)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#
#     # def get_attention_mask(self, item_seq):
#     #     """Generate left-to-right uni-directional attention mask for multi-head attention."""
#     #     attention_mask = (item_seq > 0).long()
#     #     extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#     #     # mask for left-to-right unidirectional
#     #     max_len = attention_mask.size(-1)
#     #     attn_shape = (1, max_len, max_len)
#     #     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#     #     subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#     #     subsequent_mask = subsequent_mask.long().to(item_seq.device)
#     #
#     #     extended_attention_mask = extended_attention_mask * subsequent_mask
#     #     extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#     #     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#     #     return extended_attention_mask
#     def get_attention_mask (self,  item_seq_len ):
#         """Generate left-to-right uni-directional attention mask for multi-head attention."""
#         batch_size = item_seq_len.shape[0]
#         seq_range = torch.arange(0, self.max_seq_length).long().to(item_seq_len.device)
#         seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, self.max_seq_length)
#         # print(seq_range_expand)
#
#         seq_len_expand = item_seq_len.reshape(-1, 1).repeat(1, self.max_seq_length)
#         # print(seq_len_expand)
#
#         attention_mask = (seq_range_expand<seq_len_expand).long()
#
#         # attention_mask = (new_item_seq > 0).long() # batch_size, max_len
#
#
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64 512，1，1，50
#         # mask for the whole sequence
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         # subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         # subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#         # subsequent_mask = subsequent_mask.long().to(item_seq.device) # 1，1，50，50
#         subsequent_mask = torch.ones(attn_shape).unsqueeze(0).to(item_seq_len.device)
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         return extended_attention_mask
#     def forward(self, item_seq, item_seq_len,time_seq,target_time):
#         # print(item_seq.size(1))
#         # print(item_seq.device)
#         """
#
#         :param item_seq:
#         :param item_seq_len:
#         :param time_seq: batch_size,max_seq_len
#         :param target_time: batch_size,
#         :return:
#         """
#
#
#
#         position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
#         position_embedding = self.position_embedding(position_ids)
#
#         mask_index = torch.ones(target_time.shape).long().to(item_seq.device)*self.n_items
#         new_item_seq = item_seq.scatter(1, item_seq_len.reshape(-1, 1), mask_index.reshape(-1, 1))
#
#         item_emb = self.item_embedding(new_item_seq)
#         input_emb = item_emb + position_embedding
#         input_emb = self.LayerNorm(input_emb)
#         input_emb = self.dropout(input_emb)
#
#
#         new_time_seq = time_seq.scatter(1,item_seq_len.reshape(-1,1),target_time.reshape(-1,1))
#
#         item_seq_len = item_seq_len
#
#         extended_attention_mask = self.get_attention_mask(item_seq_len) # 多用一个mask位
#
#         trm_output = self.trm_encoder(input_emb,new_time_seq, extended_attention_mask, output_all_encoded_layers=True)
#         output = trm_output[-1]
#         output = self.gather_indexes(output, item_seq_len-1)
#         return output  # [B H]
#
#
#
#
#
#     def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq,target_time)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return logits
#
#     def calculate_logits_time(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq,target_time)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         predict_intervals = self.time_linear(seq_output )
#
#         return logits, predict_intervals
#
#     def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq,target_time)
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
#         seq_output = self.forward(item_seq,item_seq_len,time_seq,target_time) # B, emb_size
#
#         actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size
#
#         actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K
#
#         return actual_noise_scores
#
# class TimePredSASRec(SequentialRecommender):
#     r"""
#     SASRec is the first sequential recommender based on self-attentive mechanism.
#     NOTE:
#         In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
#         by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
#         using Fully Connected Layer to implement the PFFN.
#     """
#
#     def __init__(self, config, dataset):
#         super(TimePredSASRec, self).__init__(config, dataset)
#
#         # load parameters info
#         self.n_layers = config['n_layers']
#         self.n_heads = config['n_heads']
#         self.hidden_size = config['hidden_size']  # same as embedding_size
#         self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
#         self.hidden_dropout_prob = config['hidden_dropout_prob']
#         self.attn_dropout_prob = config['attn_dropout_prob']
#         self.hidden_act = config['hidden_act']
#         self.layer_norm_eps = float(config['layer_norm_eps'])
#
#         self.initializer_range = config['initializer_range']
#
#         # define layers and loss
#         self.item_embedding = nn.Embedding(self.n_items+1, self.hidden_size)
#         self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
#         self.trm_encoder = TimePredTransformerEncoder(
#             n_layers=self.n_layers,
#             n_heads=self.n_heads,
#             max_seq_len= self.max_seq_length,
#             hidden_size=self.hidden_size,
#             inner_size=self.inner_size,
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             attn_dropout_prob=self.attn_dropout_prob,
#             hidden_act=self.hidden_act,
#             layer_norm_eps=self.layer_norm_eps
#         )
#
#         self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
#         self.dropout = nn.Dropout(self.hidden_dropout_prob)
#
#         # self.time_linear = nn.Linear(self.hidden_size,1)
#
#         self.relu_fun = nn.ReLU()
#
#         self.device = config['device']
#
#
#         # parameters initialization
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.initializer_range)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#
#     # def get_attention_mask(self, item_seq):
#     #     """Generate left-to-right uni-directional attention mask for multi-head attention."""
#     #     attention_mask = (item_seq > 0).long()
#     #     extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#     #     # mask for left-to-right unidirectional
#     #     max_len = attention_mask.size(-1)
#     #     attn_shape = (1, max_len, max_len)
#     #     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#     #     subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#     #     subsequent_mask = subsequent_mask.long().to(item_seq.device)
#     #
#     #     extended_attention_mask = extended_attention_mask * subsequent_mask
#     #     extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#     #     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#     #     return extended_attention_mask
#     def get_attention_mask (self,  item_seq_len ):
#         """Generate left-to-right uni-directional attention mask for multi-head attention."""
#         batch_size = item_seq_len.shape[0]
#         seq_range = torch.arange(0, self.max_seq_length).long().to(item_seq_len.device)
#         seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, self.max_seq_length)
#         # print(seq_range_expand)
#
#         seq_len_expand = item_seq_len.reshape(-1, 1).repeat(1, self.max_seq_length)
#         # print(seq_len_expand)
#
#         attention_mask = (seq_range_expand<seq_len_expand).long()
#
#         # attention_mask = (new_item_seq > 0).long() # batch_size, max_len
#
#
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64 512，1，1，50
#         # mask for the whole sequence
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         # subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         # subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#         # subsequent_mask = subsequent_mask.long().to(item_seq.device) # 1，1，50，50
#         subsequent_mask = torch.ones(attn_shape).unsqueeze(0).to(item_seq_len.device)
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         return extended_attention_mask
#
#     def get_avg_intervals(self,time_seq,item_seq_len):
#
#         time_intervals = time_seq[:, 1:] - time_seq[:, :-1]
#         zero_intervals = torch.zeros(item_seq_len.shape).reshape(-1, 1).to(time_seq.device)
#         time_intervals = torch.cat([zero_intervals, time_intervals], 1)
#         time_intervals[time_intervals < 0] = 0  # batch_size, max_seq_len
#         avg_time_intervals = torch.sum(time_intervals, 1) / item_seq_len.reshape(-1, )  # batch_size, 1
#         avg_time_intervals = avg_time_intervals.reshape(-1,1)
#
#
#         return avg_time_intervals
#
#     def forward(self, item_seq, item_seq_len,time_seq,target_time):
#         # print(item_seq.size(1))
#         # print(item_seq.device)
#         """
#
#         :param item_seq:
#         :param item_seq_len:
#         :param time_seq: batch_size,max_seq_len
#         :param target_time: batch_size,
#         :return:
#         """
#
#
#
#         position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
#         position_embedding = self.position_embedding(position_ids)
#
#         mask_index = torch.ones(target_time.shape).long().to(item_seq.device)*self.n_items
#         new_item_seq = item_seq.scatter(1, item_seq_len.reshape(-1, 1), mask_index.reshape(-1, 1))
#
#         item_emb = self.item_embedding(new_item_seq)
#         input_emb = item_emb + position_embedding
#         input_emb = self.LayerNorm(input_emb)
#         input_emb = self.dropout(input_emb)
#
#         last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
#
#         avg_time_intervals = self.get_avg_intervals(time_seq,item_seq_len)
#         coarse_target_time = last_time+avg_time_intervals
#
#         new_time_seq = time_seq.scatter(1,item_seq_len.reshape(-1,1),coarse_target_time.reshape(-1,1))
#
#
#         item_seq_len = item_seq_len+1
#
#         extended_attention_mask = self.get_attention_mask(item_seq_len) # 多用一个mask位
#
#         emb_output,interval_output = self.trm_encoder(input_emb,new_time_seq, item_seq_len,extended_attention_mask, output_all_encoded_layers=True)
#
#         emb_output = emb_output[-1]
#
#         emb_output = self.gather_indexes(emb_output, item_seq_len-1)
#         interval_output = interval_output[-1]
#
#         return emb_output,interval_output  # [B H]
#
#
#
#
#
#     def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#
#
#         seq_output,pred_intervals = self.forward(item_seq, item_seq_len,time_seq,target_time)
#
#         test_item_emb  = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return logits
#
#     def calculate_logits_time(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#
#
#         seq_output,predict_intervals = self.forward(item_seq, item_seq_len,time_seq,target_time)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#
#
#         return logits, predict_intervals
#
#     def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq,target_time)
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
#         seq_output = self.forward(item_seq,item_seq_len,time_seq,target_time) # B, emb_size
#
#         actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size
#
#         actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K
#
#         return actual_noise_scores
#
# class TimeAwareSASRec2(SequentialRecommender):
#     r"""
#     WSDM2020
#     """
#
#     def __init__(self, config, dataset):
#         super(TimeAwareSASRec2, self).__init__(config, dataset)
#
#         # load parameters info
#         self.n_layers = config['n_layers']
#         self.n_heads = config['n_heads']
#         self.hidden_size = config['hidden_size']  # same as embedding_size
#         self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
#         self.hidden_dropout_prob = config['hidden_dropout_prob']
#         self.attn_dropout_prob = config['attn_dropout_prob']
#         self.hidden_act = config['hidden_act']
#         self.layer_norm_eps = float(config['layer_norm_eps'])
#
#         self.initializer_range = config['initializer_range']
#
#         # define layers and loss
#         self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)
#         self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
#         self.trm_encoder = TimeAwareTransformerEncoder2(
#             n_layers=self.n_layers,
#             n_heads=self.n_heads,
#             max_seq_len= self.max_seq_length,
#             hidden_size=self.hidden_size,
#             inner_size=self.inner_size,
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             attn_dropout_prob=self.attn_dropout_prob,
#             hidden_act=self.hidden_act,
#             layer_norm_eps=self.layer_norm_eps
#         )
#
#         self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
#         self.dropout = nn.Dropout(self.hidden_dropout_prob)
#
#         self.device = config['device']
#
#
#         # parameters initialization
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.initializer_range)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#
#     def get_attention_mask(self, item_seq):
#         """Generate left-to-right uni-directional attention mask for multi-head attention."""
#         attention_mask = (item_seq > 0).long()
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#         # mask for left-to-right unidirectional
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#         subsequent_mask = subsequent_mask.long().to(item_seq.device)
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         return extended_attention_mask
#
#     def forward(self, item_seq, item_seq_len,time_seq):
#         # print(item_seq.size(1))
#         # print(item_seq.device)
#         position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
#         position_embedding = self.position_embedding(position_ids)
#
#         item_emb = self.item_embedding(item_seq)
#         input_emb = item_emb  # TODO 单纯 item embedding
#         input_emb = self.LayerNorm(input_emb)
#         input_emb = self.dropout(input_emb)
#
#         extended_attention_mask = self.get_attention_mask(item_seq)
#
#         trm_output = self.trm_encoder(input_emb,time_seq, extended_attention_mask, output_all_encoded_layers=True)
#         output = trm_output[-1]
#         output = self.gather_indexes(output, item_seq_len - 1)
#         return output  # [B H]
#
#     def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return logits
#
#     def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq)
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
#         seq_output = self.forward(item_seq,item_seq_len,time_seq) # B, emb_size
#
#         actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size
#
#         actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K
#
#         return actual_noise_scores
#
#
#
#
# class TimeAwareSASRec3(SequentialRecommender):
#     r"""
#     SASRec is the first sequential recommender based on self-attentive mechanism.
#     NOTE:
#         In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
#         by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
#         using Fully Connected Layer to implement the PFFN.
#     """
#
#     def __init__(self, config, dataset):
#         super(TimeAwareSASRec3, self).__init__(config, dataset)
#
#         # load parameters info
#         self.n_layers = config['n_layers']
#         self.n_heads = config['n_heads']
#         self.hidden_size = config['hidden_size']  # same as embedding_size
#         self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
#         self.hidden_dropout_prob = config['hidden_dropout_prob']
#         self.attn_dropout_prob = config['attn_dropout_prob']
#         self.hidden_act = config['hidden_act']
#         self.layer_norm_eps = float(config['layer_norm_eps'])
#
#         self.initializer_range = config['initializer_range']
#
#         # define layers and loss
#         self.item_embedding = nn.Embedding(self.n_items, self.hidden_size )
#         self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
#         self.time_embedding = nn.Embedding(100,self.hidden_size)
#         self.trm_encoder = TransformerEncoder(
#             n_layers=self.n_layers,
#             n_heads=self.n_heads,
#             hidden_size=self.hidden_size,
#             inner_size=self.inner_size,
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             attn_dropout_prob=self.attn_dropout_prob,
#             hidden_act=self.hidden_act,
#             layer_norm_eps=self.layer_norm_eps
#         )
#
#         self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
#         self.dropout = nn.Dropout(self.hidden_dropout_prob)
#
#         self.device = config['device']
#
#
#         # parameters initialization
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.initializer_range)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#
#     def get_attention_mask(self, item_seq):
#         """Generate left-to-right uni-directional attention mask for multi-head attention."""
#         attention_mask = (item_seq > 0).long()
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#         # mask for left-to-right unidirectional
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # TODO 下三角，包括对角线
#         subsequent_mask = subsequent_mask.long().to(item_seq.device)
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         # extended_attention_mask = extended_attention_mask
#
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         return extended_attention_mask
#
#     def forward(self, item_seq, item_seq_len,time_seq,time_interval_seq):
#         # print(item_seq.size(1))
#         # print(item_seq.device)
#         position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
#         position_embedding = self.position_embedding(position_ids)
#
#         # first_time = time_seq[:,0].unsqueeze(-1)
#         # diff = torch.log(1+torch.abs(time_seq-first_time)*60).long()
#         diff = torch.log(1+abs(time_interval_seq*60)).long()
#         time_embedding = self.time_embedding(diff)
#
#
#         item_emb = self.item_embedding(item_seq)
#         input_emb = item_emb + position_embedding+time_embedding # TODO 增加了time_embedding
#
#         input_emb = self.LayerNorm(input_emb)
#         input_emb = self.dropout(input_emb)
#
#         extended_attention_mask = self.get_attention_mask(item_seq)
#
#         trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
#         output = trm_output[-1]
#         output = self.gather_indexes(output, item_seq_len - 1)
#         return output  # [B H]
#
#     def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq,time_interval_seq)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return logits
#
#
#     def get_emb_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len)
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
#         seq_output = self.forward(item_seq,item_seq_len) # B, emb_size
#
#         actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size
#
#         actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K
#
#         return actual_noise_scores
#
#
#
#
#     def get_noise_score(self,item_seq,item_seq_len,target_id,K):
#         """
#
#         :param item_seq:
#         :param item_seq_len:
#         :param target_id: B*1
#         :param K:
#         :return: B*K
#         """
#         device = item_seq.device
#         softmax_fun = nn.Softmax(dim = 1)
#         logits = softmax_fun(self.calculate_logits(item_seq,item_seq_len) )
#         ones = torch.ones( logits.shape  ).to(device)
#         one_hot = ones.scatter_(1, target_id.reshape(-1,1) , 0)
#         logits *= one_hot
#
#         topk_score,topk = logits.topk(k=K)
#         return topk_score,topk
#
#     def get_target_score(self,item_seq,item_seq_len,target_id):
#         seq_output = self.forward(item_seq, item_seq_len)  # B*emb_size
#         target_emb = self.item_embedding(target_id).squeeze(1)  # B*emb_size
#         target_score = torch.sum(seq_output * target_emb, axis=1)
#         return target_score
#


# class TimeAutoRegrSASRec (SequentialRecommender):
#     r"""
#     SASRec is the first sequential recommender based on self-attentive mechanism.
#     NOTE:
#         In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
#         by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
#         using Fully Connected Layer to implement the PFFN.
#     """
#
#     def __init__(self, config, dataset):
#         super(TimeAutoRegrSASRec, self).__init__(config, dataset)
#
#         # load parameters info
#         self.n_layers = config['n_layers']
#         self.n_heads = config['n_heads']
#         self.hidden_size = config['hidden_size']  # same as embedding_size
#         self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
#         self.hidden_dropout_prob = config['hidden_dropout_prob']
#         self.attn_dropout_prob = config['attn_dropout_prob']
#         self.hidden_act = config['hidden_act']
#         self.layer_norm_eps = float(config['layer_norm_eps'])
#
#         self.initializer_range = config['initializer_range']
#
#         # define layers and loss
#         self.item_embedding = nn.Embedding(self.n_items, self.hidden_size )
#         self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
#         self.reverse_position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
#         self.time_embedding = nn.Embedding(100,self.hidden_size)
#         self.trm_encoder = TransformerEncoder(
#             n_layers=self.n_layers,
#             n_heads=self.n_heads,
#             hidden_size=self.hidden_size,
#             inner_size=self.inner_size,
#             hidden_dropout_prob=self.hidden_dropout_prob,
#             attn_dropout_prob=self.attn_dropout_prob,
#             hidden_act=self.hidden_act,
#             layer_norm_eps=self.layer_norm_eps
#         )
#         self.time_embedding = nn.Embedding(100, self.hidden_size)
#         self.gru_layers = nn.GRU(
#             input_size=self.hidden_size,
#             hidden_size=self.hidden_size
#         )
#
#         self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
#         self.dropout = nn.Dropout(self.hidden_dropout_prob)
#
#
#         self.time_linear = nn.Linear(self.hidden_size,1)
#
#         self.device = config['device']
#
#
#         # parameters initialization
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.initializer_range)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#
#     def get_attention_mask(self, item_seq):
#         """Generate left-to-right uni-directional attention mask for multi-head attention."""
#         attention_mask = (item_seq > 0).long()
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#         # mask for left-to-right unidirectional
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # TODO 下三角，包括对角线
#         subsequent_mask = subsequent_mask.long().to(item_seq.device)
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         # extended_attention_mask = extended_attention_mask
#
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         return extended_attention_mask
#
#     def forward(self, item_seq, item_seq_len,time_seq,time_interval_seq):
#         # print(item_seq.size(1))
#         # print(item_seq.device)
#         position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
#         position_embedding = self.position_embedding(position_ids)
#
#         # first_time = time_seq[:,0].unsqueeze(-1)
#         # diff = torch.log(1+torch.abs(time_seq-first_time)*60).long()
#         # diff = torch.log(1+abs(time_interval_seq*60)).long()
#         # time_embedding = self.time_embedding(diff)
#
#
#         item_emb = self.item_embedding(item_seq)
#         input_emb = item_emb + position_embedding  # TODO 增加了time_embedding
#
#         input_emb = self.LayerNorm(input_emb)
#         input_emb = self.dropout(input_emb)
#
#         extended_attention_mask = self.get_attention_mask(item_seq)
#
#         trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
#         output = trm_output[-1]
#         output = self.gather_indexes(output, item_seq_len - 1)
#         return output  # [B H]
#
#     def calculate_logits(self, item_seq, item_seq_len, time_seq,time_interval_seq,target_time):
#
#         seq_output = self.forward(item_seq, item_seq_len,time_seq,time_interval_seq)
#
#         test_item_emb = self.item_embedding.weight
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#
#         return logits
#
#
#     def get_pos_emb(self,item_seq,item_seq_len):
#         position_ids = torch.arange(item_seq.size(1),dtype = torch.long,device = item_seq.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
#         new_seq_len = item_seq_len.view(-1,1)
#         reverse_pos_id = new_seq_len - position_ids
#         reverse_pos_id =  torch.clamp(reverse_pos_id,0)
#
#         reverse_position_embedding = self.reverse_position_embedding(reverse_pos_id)
#         return reverse_position_embedding
#
#     def cal_interval_time(self,item_seq,item_seq_len,time_seq,time_interval_seq,target_time):
#
#         # TODO 待完善，其实把时间放右边比较合理
#
#
#         # reverse_pos_emb = self.get_pos_emb(time_seq,item_seq_len) # B, max_len,num_units
#         #
#         # concanated_emb = torch.cat((reverse_pos_emb,time_interval_seq.unsqueeze(-1)),-1)
#         #
#         # gru_outputs, _ = self.gru_layers(concanated_emb)
#
#         # gru_outputs, _ = self.gru_layers(time_interval_seq.unsqueeze(-1))
#
#         time_interval_seq = time_interval_seq/24
#         time_interval_seq[time_interval_seq>30] = 30 # 把最大值限制为30
#
#         diff = abs(time_interval_seq)
#
#         # diff = torch.log(1 + abs(time_interval_seq )).long()
#         time_embedding = self.time_embedding(diff.long())
#         item_emb = self.item_embedding(item_seq)
#         gru_inputs = time_embedding+item_emb
#         gru_outputs, _ = self.gru_layers(gru_inputs)
#
#
#         output_emb = self.gather_indexes(gru_outputs, item_seq_len - 1) # B, num_units
#
#         predict_intervals = self.time_linear(output_emb)
#
#
#         fun = nn.ReLU()
#         predict_intervals = fun(predict_intervals)
#
#
#
#
#
#         return predict_intervals
#
#
#
#
#
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
#         seq_output = self.forward(item_seq, item_seq_len,time_seq,time_interval_seq) # B, emb_size
#
#         actual_noise_emb = self.item_embedding(pos_neg_items) # B, 1+K, emb_size
#
#         actual_noise_scores = torch.sum(seq_output.unsqueeze(1)*actual_noise_emb,2) # B, 1+K
#
#         return actual_noise_scores
#
#
#
#
#     def get_noise_score(self,item_seq,item_seq_len,target_id,K):
#         """
#
#         :param item_seq:
#         :param item_seq_len:
#         :param target_id: B*1
#         :param K:
#         :return: B*K
#         """
#         device = item_seq.device
#         softmax_fun = nn.Softmax(dim = 1)
#         logits = softmax_fun(self.calculate_logits(item_seq,item_seq_len) )
#         ones = torch.ones( logits.shape  ).to(device)
#         one_hot = ones.scatter_(1, target_id.reshape(-1,1) , 0)
#         logits *= one_hot
#
#         topk_score,topk = logits.topk(k=K)
#         return topk_score,topk
#
#     def get_target_score(self,item_seq,item_seq_len,target_id):
#         seq_output = self.forward(item_seq, item_seq_len)  # B*emb_size
#         target_emb = self.item_embedding(target_id).squeeze(1)  # B*emb_size
#         target_score = torch.sum(seq_output * target_emb, axis=1)
#         return target_score