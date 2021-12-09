# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 20:26:07 2021

@author: Meghana
"""
import dgl
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch


class OneConvTwoClassiLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(OneConvTwoClassiLayerGCN, self).__init__()
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


class TwoConvThreeClassiLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim2, n_classes):
        super(TwoConvThreeClassiLayerGCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim2)
        self.classify = nn.Sequential(nn.Linear(hidden_dim2, 12),nn.Linear(12, 12),
                                      nn.Linear(12, n_classes))

    def forward(self, g, inputs):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1.forward(g, inputs))
        h = F.relu(self.conv2.forward(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')    # Equivalent to dgl.readout_nodes(g, 'h', op='mean')
            return self.classify(hg)
        
        
class TwoConvFourClassiLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim2, n_classes):
        super(TwoConvFourClassiLayerGCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim2)
        self.classify = nn.Sequential(nn.Linear(hidden_dim2, 18),nn.Linear(18, 12),nn.Linear(12, 6),
                                      nn.Linear(6, n_classes))

    def forward(self, g, inputs):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1.forward(g, inputs))
        h = F.relu(self.conv2.forward(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')    # Equivalent to dgl.readout_nodes(g, 'h', op='mean')
            return self.classify(hg)
        

class OneConvSAGE(nn.Module):
    """Graph convolution module used by the GraphSAGE model.
    
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(OneConvSAGE, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)
    
    def forward(self, g, h):
        """Forward computation
        
        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        # All the `ndata` set within a local scope will be automatically popped out.
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            h_total = torch.cat([h, h_neigh], dim=1)
            return F.relu(self.linear(h_total))        
        

class SAGEOneConvLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SAGEOneConvLayer, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_dim*2, hidden_dim),
                                    nn.Linear(hidden_dim, out_dim))

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            
            # update_all is a message passing API.
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            h_total = torch.cat([h, h_neigh], dim=1)
            return F.relu(self.linear(h_total))
        
        
class TwoConvSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(TwoConvSAGE, self).__init__()
        self.conv1 = SAGEOneConvLayer(in_dim, hidden_dim, 12)        # 12 chosen randomly
        self.conv2 = SAGEOneConvLayer(12, hidden_dim, 12)
        self.classify = nn.Linear(12, n_classes)
       
    def forward(self, g, inputs):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1.forward(g, inputs))
        h = F.relu(self.conv2.forward(g, h))
        
        with g.local_scope():
            g.ndata['h'] = h
            
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')    # Equivalent to dgl.readout_nodes(g, 'h', op='mean')
            return self.classify(hg)        