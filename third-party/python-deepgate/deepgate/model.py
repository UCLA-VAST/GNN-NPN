from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch.nn import LSTM, GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph
from .utils.utils import generate_hs_init

from .arch.mlp import MLP
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv

class Model(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self,
                 num_rounds = 1,
                 dim_hidden = 128,
                 enable_encode = False,
                 enable_reverse = False,
                 enable_layernorm = False
                ):
        super(Model, self).__init__()

        # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse
        self.enable_layernorm = enable_layernorm

        # dimensions
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        # Network 
        self.aggr_and = TFMlpAggr(self.dim_hidden, self.dim_hidden)
        self.aggr_not = TFMlpAggr(self.dim_hidden, self.dim_hidden)
        if self.enable_reverse:
            self.aggr_and_rev = TFMlpAggr(self.dim_hidden, self.dim_hidden, reverse=True)
            self.aggr_not_rev = TFMlpAggr(self.dim_hidden, self.dim_hidden, reverse=True)
            self.aggr_pi_rev = TFMlpAggr(self.dim_hidden, self.dim_hidden, reverse=True)

        self.update_and = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not = GRU(self.dim_hidden, self.dim_hidden)
        if self.enable_reverse:
            self.update_and_rev = GRU(self.dim_hidden, self.dim_hidden)
            self.update_not_rev = GRU(self.dim_hidden, self.dim_hidden)
            self.update_pi_rev = GRU(self.dim_hidden, self.dim_hidden)

        if self.enable_layernorm:
            self.norm = nn.LayerNorm(self.dim_hidden)

        # Readout
        self.readout_out_neg = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0, norm_layer='layernorm', act_layer='relu')
        self.readout_in_neg = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0, norm_layer='layernorm', act_layer='relu')
        self.readout_in_perm = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0, norm_layer='layernorm', act_layer='relu')

        # # consider the embedding for the LSTM/GRU model initialized by non-zeros
        # self.one = torch.ones(1)
        # # self.hs_emd_int = nn.Linear(1, self.dim_hidden)
        # self.hf_emd_int = nn.Linear(1, self.dim_hidden)
        # self.one.requires_grad = False

    def forward(self, G):
        device = next(self.parameters()).device
        num_nodes = len(G.gate)
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1

        # initialize node state
        node_state = torch.zeros(num_nodes, self.dim_hidden)
        if self.enable_encode:
            node_state = generate_hs_init(G, node_state, self.dim_hidden)
        node_state = node_state.to(device)

        edge_index = G.edge_index

        pi_mask = G.gate.squeeze(1) == 0
        and_mask = G.gate.squeeze(1) == 1
        not_mask = G.gate.squeeze(1) == 2

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == level

                # AND Gate
                l_and_node = G.forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, dim=1)
                    msg = self.aggr_and(node_state, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    node_state_and = torch.index_select(node_state, dim=0, index=l_and_node)
                    _, node_state_and = self.update_and(and_msg.unsqueeze(0), node_state_and.unsqueeze(0))
                    node_state[l_and_node, :] = node_state_and.squeeze(0)

                # NOT Gate
                l_not_node = G.forward_index[layer_mask & not_mask]
                if l_not_node.size(0) > 0:
                    not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, dim=1)
                    msg = self.aggr_not(node_state, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    node_state_not = torch.index_select(node_state, dim=0, index=l_not_node)
                    _, node_state_not = self.update_not(not_msg.unsqueeze(0), node_state_not.unsqueeze(0))
                    node_state[l_not_node, :] = node_state_not.squeeze(0)

            if self.enable_reverse:
                for level in range(1, num_layers_b):
                    # backward layer
                    layer_mask = G.backward_level == level

                    # AND Gate
                    l_and_node = G.backward_index[layer_mask & and_mask]
                    if l_and_node.size(0) > 0:
                        and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, dim=0)
                        msg = self.aggr_and_rev(node_state, and_edge_index, and_edge_attr)
                        and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                        node_state_and = torch.index_select(node_state, dim=0, index=l_and_node)
                        _, node_state_and = self.update_and_rev(and_msg.unsqueeze(0), node_state_and.unsqueeze(0))
                        node_state[l_and_node, :] = node_state_and.squeeze(0)

                    # NOT Gate
                    l_not_node = G.backward_index[layer_mask & not_mask]
                    if l_not_node.size(0) > 0:
                        not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, dim=0)
                        msg = self.aggr_not_rev(node_state, not_edge_index, not_edge_attr)
                        not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                        node_state_not = torch.index_select(node_state, dim=0, index=l_not_node)
                        _, node_state_not = self.update_not_rev(not_msg.unsqueeze(0), node_state_not.unsqueeze(0))
                        node_state[l_not_node, :] = node_state_not.squeeze(0)

                    # PI Node
                    l_pi_node = G.backward_index[layer_mask & pi_mask]
                    if l_pi_node.size(0) > 0:
                        pi_edge_index, pi_edge_attr = subgraph(l_pi_node, edge_index, dim=0)
                        msg = self.aggr_pi_rev(node_state, pi_edge_index, pi_edge_attr)
                        pi_msg = torch.index_select(msg, dim=0, index=l_pi_node)
                        node_state_pi = torch.index_select(node_state, dim=0, index=l_pi_node)
                        _, node_state_pi = self.update_pi_rev(pi_msg.unsqueeze(0), node_state_pi.unsqueeze(0))
                        node_state[l_pi_node, :] = node_state_pi.squeeze(0)

            if self.enable_layernorm:
                node_state = self.norm(node_state)

        return node_state

    def pred_out_neg(self, node_state):
        logits = self.readout_out_neg(node_state)
        return logits

    def pred_in_neg(self, node_state):
        logits = self.readout_in_neg(node_state)
        return logits

    def pred_in_perm(self, node_state):
        logits = self.readout_in_perm(node_state)
        return logits

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        
    def load_pretrained(self, pretrained_model_path = ''):
        if pretrained_model_path == '':
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)
