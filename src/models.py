# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:26:07 2021

@author: Meghana
"""
import dgl
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class SimpleGCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(SimpleGCNClassifier, self).__init__()
        self.conv = dglnn.GraphConv(in_dim, hidden_dim)
        self.classify = nn.Sequential(nn.Linear(hidden_dim, 12),
                                      nn.Linear(12, n_classes))

    def forward(self, g, inputs):
        # Apply graph convolution and activation.
        h = F.relu(self.conv.forward(g, inputs))
        with g.local_scope():
            g.ndata['h'] = h
            
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')    # Equivalent to dgl.readout_nodes(g, 'h', op='mean')
            return self.classify(hg)