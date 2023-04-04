# -*- coding: utf-8 -*-

def gather_indexes( output, gather_index):
    """Gathers the vectors at the spexific positions over a minibatch"""
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)