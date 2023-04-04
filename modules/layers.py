# -*- coding: utf-8 -*-

import copy
import math

import torch
import torch.nn as nn

import torch.nn.functional as fn

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.
    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer
    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer
    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.
        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers



class TimeAwareMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(self, n_heads, max_seq_len,hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(TimeAwareMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        """
        for time decay
        """
        self.max_seq_len = max_seq_len
        self.time_query_layer = nn.Linear(hidden_size,self.all_head_size)
        self.time_input_w1 = nn.Parameter(torch.zeros (self.max_seq_len, self.max_seq_len))
        self.time_input_b = nn.Parameter(torch.zeros (self.max_seq_len, self.max_seq_len))

        self.time_output_w1 = nn.Parameter(torch.zeros (self.max_seq_len,self.max_seq_len))
        self.time_output_w2 = nn.Parameter(torch.zeros (self.max_seq_len, self.max_seq_len))
        self.time_output_b = nn.Parameter(torch.zeros (self.max_seq_len, self.max_seq_len))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # B, seq_len, n_head, head_size
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # B, n_head,seq_len, head_size

    def forward(self, input_tensor, time_seq,attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        """
        time decay
        """

        time_query_key = self.time_query_layer(input_tensor) # B, seq_len, num_items
        time_query_key = torch.matmul(time_query_key,input_tensor.transpose(-1,-2))
        time_query_key = torch.tanh(time_query_key)

        batch_size, max_len = time_seq.size()

        time_intervals = time_seq.repeat(1,max_len)
        time_intervals = time_intervals.reshape(batch_size,max_len,max_len)

        t_query = time_intervals.permute(0,2,1)
        t_key = time_intervals

        diff = torch.abs(t_query-t_key)
        # TODO 是不是log这里出现的问题
        decay = torch.log(1+diff )
        # decay = diff

        # tmp = torch.sum((decay==0).long())
        # if tmp.data>0:
        #     print(tmp)
        #     print('find some vale less than 0 in decay-----------------------------')

        decay = torch.tanh(decay*self.time_input_w1+self.time_input_b)

        decay_gate = self.time_output_w1*decay + self.time_output_w2*time_query_key+self.time_output_b

        decay_gate_ = decay_gate.repeat(self.num_attention_heads,1,1).reshape(batch_size,self.num_attention_heads,self.max_seq_len,self.max_seq_len)

        time_attention_scores = torch.sigmoid(decay_gate_)

        attention_scores = attention_scores*time_attention_scores
        # attention_scores = attention_scores + time_attention_scores

        # scale
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TimeAwareTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
            self, n_heads, max_seq_len,hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps
    ):
        super(TimeAwareTransformerLayer, self).__init__()
        self.multi_head_attention = TimeAwareMultiHeadAttention(
            n_heads, max_seq_len,hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, time_seq,attention_mask):
        attention_output = self.multi_head_attention(hidden_states, time_seq,attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TimeAwareTransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.
        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            max_seq_len = 50,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12
    ):

        super(TimeAwareTransformerEncoder, self).__init__()
        layer = TimeAwareTransformerLayer(
            n_heads, max_seq_len,hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, time_seq,attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, time_seq,attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class TimePredMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(self, n_heads, max_seq_len,hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(TimePredMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        """
        for time decay
        """
        self.max_seq_len = max_seq_len
        self.time_query_layer = nn.Linear(hidden_size,self.all_head_size)
        self.time_input_w1 = nn.Parameter(torch.zeros (self.max_seq_len, self.max_seq_len))
        self.time_input_b = nn.Parameter(torch.zeros (self.max_seq_len, self.max_seq_len))

        self.time_output_w1 = nn.Parameter(torch.zeros (self.max_seq_len,self.max_seq_len))
        self.time_output_w2 = nn.Parameter(torch.zeros (self.max_seq_len, self.max_seq_len))
        self.time_output_b = nn.Parameter(torch.zeros (self.max_seq_len, self.max_seq_len))

        """
        for time predict
        """
        self.time_linear = nn.Linear( hidden_size*2 ,1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # B, seq_len, n_head, head_size
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # B, n_head,seq_len, head_size

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, input_tensor, time_seq,item_seq_len,attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        """
        time decay
        """

        time_query_key = self.time_query_layer(input_tensor) # B, seq_len, num_items
        time_query_key = torch.matmul(time_query_key,input_tensor.transpose(-1,-2))
        time_query_key = torch.tanh(time_query_key)

        batch_size, max_len = time_seq.size()

        time_intervals = time_seq.repeat(1,max_len)
        time_intervals = time_intervals.reshape(batch_size,max_len,max_len)

        t_query = time_intervals.permute(0,2,1)
        t_key = time_intervals

        diff = torch.abs(t_query-t_key)
        # TODO 是不是log这里出现的问题
        decay = torch.log(1+diff )
        # decay = diff

        # tmp = torch.sum((decay==0).long())
        # if tmp.data>0:
        #     print(tmp)
        #     print('find some vale less than 0 in decay-----------------------------')

        decay = torch.tanh(decay*self.time_input_w1+self.time_input_b)

        decay_gate = self.time_output_w1*decay + self.time_output_w2*time_query_key+self.time_output_b

        decay_gate_ = decay_gate.repeat(self.num_attention_heads,1,1).reshape(batch_size,self.num_attention_heads,self.max_seq_len,self.max_seq_len)

        time_attention_scores = torch.sigmoid(decay_gate_)

        attention_scores = attention_scores*time_attention_scores
        # attention_scores = attention_scores + time_attention_scores

        # scale
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # 进行时间预测
        # time_intervals = time_seq[:, 1:] - time_seq[:, :-1]
        # zero_intervals = torch.zeros(time_seq.shape[0],1).to(time_intervals.device)
        # time_intervals = torch.cat([zero_intervals, time_intervals], 1)
        # time_intervals[time_intervals < 0] = 0  # batch_size, max_seq_len
        #
        # batch_size, num_heads,max_len,max_len = attention_scores.shape
        # flatten_attention_scores = attention_scores.reshape(-1,max_len,max_len) # B*num_heads,max_len,max_len
        # item_seq_len = item_seq_len.reshape(-1,1).repeat(1,num_heads).reshape(-1,) # B*num_heads,
        #
        # target_attention_scores = self.gather_indexes(flatten_attention_scores, item_seq_len-1) # B*num_heads, max_seq_len
        #
        # time_intervals = time_intervals.repeat(1,num_heads).reshape(-1,max_len) # B*num_heads, max_seq_len
        #
        # pred_intervals = time_intervals*target_attention_scores
        # pred_intervals = torch.sum(pred_intervals,1).reshape(batch_size,num_heads) # B,num_heads
        #
        # pred_intervals = self.time_linear(pred_intervals)

        emb1 = self.gather_indexes(hidden_states,item_seq_len-1)
        emb2 = self.gather_indexes(hidden_states, item_seq_len -2)


        pred_intervals = self.time_linear(torch.cat([emb1,emb2],1))
        # pred_intervals = self.time_linear(emb1+emb2)

        return hidden_states,pred_intervals


class TimePredTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
            self, n_heads, max_seq_len,hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps
    ):
        super(TimePredTransformerLayer, self).__init__()
        self.multi_head_attention = TimePredMultiHeadAttention(
            n_heads, max_seq_len,hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, time_seq,item_seq_len,attention_mask):
        attention_output,pred_intervals = self.multi_head_attention(hidden_states, time_seq,item_seq_len,attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output,pred_intervals


class TimePredTransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.
        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            max_seq_len = 50,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12
    ):

        super(TimePredTransformerEncoder, self).__init__()
        layer = TimePredTransformerLayer(
            n_heads, max_seq_len,hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, time_seq, item_seq_len, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        all_intervals = []

        pred_target_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 1)  # B, 1
        last_time = torch.gather(time_seq, 1, item_seq_len.reshape(-1, 1).long() - 2)

        for layer_module in self.layer:


            # pred_interval B,1
            hidden_states,pred_intervals = layer_module(hidden_states, time_seq,item_seq_len,attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_intervals.append(pred_intervals)
            # 更新time_seq
            coarse_target_time = last_time+pred_intervals
            time_seq = time_seq.scatter(1, item_seq_len.reshape(-1, 1)-1, coarse_target_time)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_intervals.append(pred_intervals)
        return all_encoder_layers,all_intervals



class TimeAwareMultiHeadAttention2(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(self, n_heads, max_seq_len,hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps ,
                 key_time_embedding_table,value_time_embedding_table,key_pos_embedding_table,value_pos_embedding_table):
        super(TimeAwareMultiHeadAttention2, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        """
        for time decay
        """
        self.key_time_embedding_table = key_time_embedding_table
        self.value_time_embedding_table = value_time_embedding_table
        self.key_pos_embedding_table = key_pos_embedding_table
        self.value_pos_embedding_table = value_pos_embedding_table


        self.key_time_linear = nn.Linear(1,self.all_head_size)

        self.value_time_linear = nn.Linear(1, self.all_head_size)

        self.max_seq_len = max_seq_len
        self.time_query_layer = nn.Linear(hidden_size,self.all_head_size)
        self.time_input_w1 = nn.Parameter(torch.Tensor(self.max_seq_len, self.max_seq_len))
        self.time_input_b = nn.Parameter(torch.Tensor(self.max_seq_len, self.max_seq_len))

        self.time_output_w1 = nn.Parameter(torch.Tensor(self.max_seq_len,self.max_seq_len))
        self.time_output_w2 = nn.Parameter(torch.Tensor(self.max_seq_len, self.max_seq_len))
        self.time_output_b = nn.Parameter(torch.Tensor(self.max_seq_len, self.max_seq_len))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # B, seq_len, n_head, head_size
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # B, n_head,seq_len, head_size

    def forward(self, input_tensor, time_seq,attention_mask ):

        position_ids = torch.arange(time_seq.size(1), dtype=torch.long, device=time_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(time_seq)

        key_pos_emb = self.key_pos_embedding_table(position_ids)
        value_pos_emb = self.value_pos_embedding_table(position_ids)


        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        key_pos_layer = self.transpose_for_scores(key_pos_emb)
        value_pos_layer = self.transpose_for_scores(value_pos_emb)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = torch.matmul(query_layer, (key_layer + key_pos_layer).transpose(-1, -2))

        """
        time decay
        """
        batch_size, max_len = time_seq.size()
        time_intervals = time_seq.repeat(1, max_len)
        time_intervals = time_intervals.reshape(batch_size, max_len, max_len)
        t_query = time_intervals.permute(0, 2, 1)
        t_key = time_intervals

        diff =  (torch.log(1+torch.abs(t_query - t_key))).long()

        key_time_emb = self.key_time_embedding_table(diff)
        # key_time_emb = self.key_time_linear(diff.unsqueeze(-1)) # B, max_len,max_len, emb_size
        key_time_emb = key_time_emb.reshape(batch_size,max_len*max_len,self.all_head_size)
        key_time_layer = self.transpose_for_scores(key_time_emb) # batch_size, n_head, max_len*max_len,head_size
        # batch_size, n_head, max_len,max_len,head_size
        key_time_layer = key_time_layer.reshape(batch_size,self.num_attention_heads,max_len,max_len,self.attention_head_size)

        value_time_emb = self.value_time_embedding_table(diff)
        # value_time_emb = self.value_time_linear(diff.unsqueeze(-1))  # B, max_len,max_len, emb_size
        value_time_emb = value_time_emb.reshape(batch_size, max_len * max_len, self.all_head_size)
        value_time_layer = self.transpose_for_scores(value_time_emb)  # batch_size, n_head, max_len*max_len,head_size
        # batch_size, n_head, max_len,max_len,head_size
        value_time_layer = value_time_layer.reshape(batch_size, self.num_attention_heads, max_len, max_len,
                                        self.attention_head_size)

        time_attention_scores = query_layer.unsqueeze(-2)*key_time_layer # batch_size, n_head,max_len,max_len,head_size
        time_attention_scores = torch.sum(time_attention_scores,-1)

        """
        time decay end
        """


        attention_scores = attention_scores+time_attention_scores







        # scale
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer+value_pos_layer) # batch_size,n_head,max_len,head_num

        # TODO for time embedding
        time_context_layer = torch.unsqueeze(attention_scores,-1)*value_time_layer # batch_size, n_head,max_len, max_len,n_head
        time_context_layer = torch.sum(time_context_layer,-2)
        context_layer = context_layer+time_context_layer # batch_size,n_head,max_len,head_num




        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TimeAwareTransformerLayer2(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
            self, n_heads, max_seq_len,hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
            layer_norm_eps,key_time_embedding_table,value_time_embedding_table,key_pos_embedding_table,value_pos_embedding_table
    ):
        super(TimeAwareTransformerLayer2, self).__init__()
        self.multi_head_attention = TimeAwareMultiHeadAttention2(
            n_heads, max_seq_len,hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,
        key_time_embedding_table,value_time_embedding_table,key_pos_embedding_table,value_pos_embedding_table)
        # self.time_embedding_talbe = time_embedding_table
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, time_seq,attention_mask ):
        attention_output = self.multi_head_attention(hidden_states, time_seq,attention_mask )
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TimeAwareTransformerEncoder2(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.
        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            max_seq_len = 50,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12
    ):

        super(TimeAwareTransformerEncoder2, self).__init__()
        self.key_time_embedding_table = nn.Embedding(100,  hidden_size)
        self.value_time_embedding_table = nn.Embedding(100,  hidden_size)

        self.key_pos_embedding_table = nn.Embedding(max_seq_len,hidden_size)
        self.value_pos_embedding_table = nn.Embedding(max_seq_len,hidden_size)

        layer = TimeAwareTransformerLayer2(
            n_heads, max_seq_len,hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
            self.key_time_embedding_table,self.value_time_embedding_table,self.key_pos_embedding_table,self.value_pos_embedding_table)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])


    def forward(self, hidden_states, time_seq,attention_mask,  output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, time_seq,attention_mask )
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

