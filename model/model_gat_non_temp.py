import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import os
import time
from torch_geometric.nn import GATConv, GATv2Conv
if os.getcwd().split('/')[2] == 'mehdi':
    from PColor import ColorPrint

    print = ColorPrint().print


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False
        self.sft = shift

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale
        #        print(f'{torch.isnan(input_).all()} num_units: {self.n_units} shift: {self.sft} ======','green')
        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[
            -1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."
        input_ = input_.reshape(-1, self.n_units)

        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False

#
# class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
#     def __init__(self, emb_size=64):
#         super().__init__('add')
#
#         self.feature_module_left = torch.nn.Sequential(
#             torch.nn.Linear(emb_size, emb_size)
#         )
#
#         self.feature_module_edge = torch.nn.Sequential(
#             torch.nn.Linear(1, emb_size, bias=False)
#         )
#         self.feature_module_right = torch.nn.Sequential(
#             torch.nn.Linear(emb_size, emb_size, bias=False)
#         )
#         self.feature_module_final = torch.nn.Sequential(
#             PreNormLayer(1, shift=False),  # removed do not remember why
#             torch.nn.ReLU(),
#             torch.nn.Linear(emb_size, emb_size)
#         )
#
#         self.post_conv_module = torch.nn.Sequential(
#             PreNormLayer(1, shift=False)
#         )
#
#         # output_layers
#         self.output_module = torch.nn.Sequential(
#             torch.nn.Linear(2 * emb_size, emb_size),
#             torch.nn.ReLU(),
#             torch.nn.Linear(emb_size, emb_size),
#         )
#
#     def forward(self, left_features, edge_indices, edge_features, right_features):
#         output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
#                                 node_features=(left_features, right_features), edge_features=edge_features)
#         return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))
#
#     def message(self, node_features_i, node_features_j, edge_features):
#         output = self.feature_module_final(self.feature_module_left(node_features_i)
#                                            + self.feature_module_edge(edge_features)
#                                            + self.feature_module_right(node_features_j))
#         return output
#

class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """

    def pre_train_init(self):
        if isinstance(self, torch.nn.parallel.DistributedDataParallel):
            self = self.module

        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        if isinstance(self, torch.nn.parallel.DistributedDataParallel):
            self = self.module
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True



class GNNPolicy(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.emb_size = args.emb_size
        self.args = args

        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        config = args.gat_config

        # CONSTRAINT EMBEDDING GOOD CANDIDATE TO REDUCE SIZE
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
        #EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
            torch.nn.Linear(edge_nfeats, self.emb_size),
            torch.nn.ReLU(),
        )

        # VARIABLE EMBEDDING   GOOD CANDIDATE TO REDUCE SIZE
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )

        if args.pytorch_gat_I:
            self.gat_v_to_c = GATConv(in_channels=(self.emb_size, self.emb_size), out_channels = self.emb_size,
                                      heads = config['num_heads_per_layer'][0], edge_dim = self.emb_size, add_self_loops = False,
                                      bias = config['bias'], dropout = config['dropout'])

            self.gat_c_to_v = GATConv(in_channels=(self.emb_size * config['num_heads_per_layer'][0],  self.emb_size), out_channels = self.emb_size,
                                      heads=config['num_heads_per_layer'][0], edge_dim=self.emb_size, add_self_loops=False,
                                      bias = config['bias'], dropout = config['dropout'])
        elif args.pytorch_gat_II:
            self.gat_v_to_c = GATv2Conv(in_channels=(self.emb_size, self.emb_size), out_channels = self.emb_size,
                                      heads = config['num_heads_per_layer'][0], edge_dim = self.emb_size, add_self_loops = False,
                                      bias = config['bias'], dropout = config['dropout'])

            self.gat_c_to_v = GATv2Conv(in_channels=(self.emb_size * config['num_heads_per_layer'][0],  self.emb_size), out_channels = self.emb_size,
                                      heads=config['num_heads_per_layer'][0], edge_dim=self.emb_size, add_self_loops=False,
                                      bias = config['bias'], dropout = config['dropout'])

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(args.emb_size * args.num_heads_per_layer[0], args.emb_size * args.num_heads_per_layer[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(args.emb_size * args.num_heads_per_layer[0], 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        variable_features = torch.nan_to_num(variable_features, 0.0)
        cons_length = constraint_features.shape[0]
        var_length = variable_features.shape[0]
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        constraint_features = self.gat_v_to_c((variable_features, constraint_features), reversed_edge_indices, edge_features)#, size=(variable_features.shape[0],constraint_features.shape[0]))

        variable_features   = self.gat_c_to_v((constraint_features, variable_features), edge_indices, edge_features)#, size=(constraint_features.shape[0], variable_features.shape[0]))

        output = self.output_module(variable_features).squeeze(-1)

        return output
