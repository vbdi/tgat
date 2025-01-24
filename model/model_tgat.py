import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import os
import time
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

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




class BipartiteGraphConvolution(MessagePassing):
    _alpha: OptTensor

    def __init__(self,args,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,):

        self.emb_size = args.emb_size
        self.gru_temp_dim = args.GRU_temp_dim
        self.gru_input_dim = args.GRU_input_dim
        self.gru_hidden_dim = args.GRU_hidden_dim
        self.args = args
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights


        #
        # self.feature_module_left = torch.nn.Sequential(
        #     torch.nn.Linear(self.emb_size, self.emb_size)
        # )
        # self.feature_module_edge = torch.nn.Sequential(
        #     torch.nn.Linear(1, self.emb_size, bias=False)
        # )
        # self.feature_module_right = torch.nn.Sequential(
        #     torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        # )
        # self.feature_module_final = torch.nn.Sequential(
        #     PreNormLayer(1, shift=False),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.emb_size, self.emb_size)
        # )
        if self.concat:
            self.post_conv_module = torch.nn.Sequential(
                PreNormLayer(1, shift=False), torch.nn.Linear(self.heads * self.emb_size, self.heads * self.emb_size)
            )
            # output_layers
            self.output_module = torch.nn.Sequential(
                torch.nn.Linear(2 * self.heads  * self.emb_size, self.emb_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_size, self.heads * out_channels),
            )

        else:
            self.post_conv_module = torch.nn.Sequential(
                PreNormLayer(1, shift=False), torch.nn.Linear(self.emb_size, self.heads * self.emb_size)
            )

            # output_layers
            self.output_module = torch.nn.Sequential(
                torch.nn.Linear(2 * self.heads  * self.emb_size, self.emb_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_size,  out_channels),
            )


        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
        N = x_r.shape[0]
        out = self.output_module(torch.cat([self.post_conv_module(out), x_r.view(N,-1)], dim=-1))
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        z = x_j * alpha.unsqueeze(-1)
        return z

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class GNNPolicy(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.emb_size = args.emb_size
        self.gru_temp_dim = args.GRU_temp_dim
        self.gru_input_dim = args.GRU_input_dim
        self.gru_hidden_dim = args.GRU_hidden_dim
        self.args = args

        bidirectional = args.GRU_bidirectional
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        self.config = args.gat_config

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
        self.elu =torch.nn.ELU()
        if self.config['concat']:
            H = self.config['num_heads_per_layer'][0]
        else:
            H = 1

        self.gat_v_to_c=BipartiteGraphConvolution(args=args,in_channels=(self.emb_size, self.emb_size), out_channels=self.emb_size,
                                            heads=self.config['num_heads_per_layer'][0], edge_dim=self.emb_size,
                                            add_self_loops=False,
                                            bias=self.config['bias'], dropout=self.config['dropout'],
                                            fill_value=self.config['fill_value'],
                                            share_weights=self.config['share_weights'], concat=self.config['concat'])
        self.gat_c_to_v=BipartiteGraphConvolution(args=args,in_channels=(self.emb_size * H, self.emb_size),
                    out_channels=self.emb_size,
                    heads=self.config['num_heads_per_layer'][0], edge_dim=self.emb_size,
                    add_self_loops=False,
                    bias=self.config['bias'], dropout=self.config['dropout'], fill_value=self.config['fill_value'],
                    share_weights=self.config['share_weights'], concat=self.config['concat'])

        self.D = 1 if bidirectional == False else 2
        self.gru = torch.nn.GRU(input_size=args.GRU_input_dim, hidden_size=args.GRU_hidden_dim,
                                num_layers=args.GRU_num_layers, bidirectional=bidirectional)

        self.gru_output_module = torch.nn.Sequential(
            torch.nn.Linear(self.gru_hidden_dim * self.D, self.gru_temp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.gru_temp_dim, 1, bias=False),
        )

    def gru_fnc(self, variable_features):
        gru_output, gru_hidden = self.gru(variable_features)
        output = self.gru_output_module(gru_output.view(-1, self.gru_hidden_dim * self.D)).squeeze(-1)
        return output

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        variable_features = torch.nan_to_num(variable_features, 0.0)
        cons_length = constraint_features.shape[0]
        var_length = variable_features.shape[0]
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        constraint_features = self.gat_v_to_c((variable_features, constraint_features), reversed_edge_indices, edge_features)
        variable_features = self.gat_c_to_v((constraint_features, variable_features), edge_indices, edge_features)

        return variable_features
