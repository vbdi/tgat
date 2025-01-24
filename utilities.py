import gzip
import pickle
import datetime
import numpy as np
import argparse
import ecole as ec
import pathlib
from environments import Branching as Environment
from rewards import TimeLimitDualIntegral as BoundIntegral
import csv
import json
import time
import math
import random
import itertools
import os
import glob
import torch
import torch.nn.functional as F
import torch_geometric
import torch.distributed as dist
from torch_geometric.data import Data, HeteroData, Dataset, Batch
from collections.abc import Mapping, Sequence

from model.model_tgat import GNNPolicy as TGATPolicy
from model.model import GNNPolicy
from model.model_gat_non_temp import GNNPolicy as GATNonTempPolicy

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def create_sequence():
    return


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            f.write(str)


def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2 ** 32 - 1:
        raise NotImplementedError
    return seed


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())

    output = torch.stack([F.pad(slice_, (0, max_pad_size - slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)

    return output


def pad_tensor_modified(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())

    output = torch.stack([F.pad(slice_, (0, max_pad_size - slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)

    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch = +1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.max_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


class ObservationFunction(ec.observation.NodeBipartite):
    def __init__(self, problem=None):
        super().__init__()

    def seed(self, seed):
        pass


class Psebservation(ec.observation.Pseudocosts):
    def __init__(self, problem=None):
        super().__init__()

    def seed(self, seed):
        pass


class FSBObservation(ec.observation.StrongBranchingScores):
    def __init__(self, problem=None):
        super().__init__()

    def seed(self, seed):
        pass


class General:
    def __init__(self, problem=None):
        self.rng = np.random.RandomState()

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        # mask variable features (no incumbent info)
        return action_set[observation[action_set].argmax()]


class Random:
    def __init__(self, problem=None):
        self.rng = np.random.RandomState()

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation=None):
        # mask variable features (no incumbent info)
        return np.random.choice(action_set)


class ExploreThenStrongBranch:
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ec.observation.Pseudocosts()
        self.strong_branching_function = ec.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        probabilities = [1 - self.expert_probability, self.expert_probability]

        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, nb_candidates, candidate_choice, candidate_scores, logits=None):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores
        self.logits = logits
        if constraint_features is not None:
            self.device = torch.device(
                'cpu') if constraint_features.get_device() == -1 else constraint_features.get_device()

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            try:
                if isinstance(self.device, list):
                    self.device = self.device[0]
                return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]]).to(
                    self.device)
            except Exception as e:
                print(f'Error: {e}', 'red')
                print(torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]]), 'blue')
                print(self.device, 'green')

                return
        elif key == 'candidates':
            return self.variable_features.size(0)
        # elif key == 'logits':
        #     print(self.logits,'---------------','green')
        #     return self.logits
        else:
            return super().__inc__(key, value, *args, **kwargs)


# Datasets
class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files_by_index = sample_files

    def len(self):
        return len(self.sample_files_by_index)

    def get(self, index):
        with gzip.open(self.sample_files_by_index[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, _, sample_action_set, sample_scores = sample['data']
        sample_scores = np.nan_to_num(sample_scores)
        sample_action = sample_action_set[sample_scores[sample_action_set].argmax()]

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, len(candidates), candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

        return graph


class GraphDatasetModified(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files_by_index = sample_files

    def len(self):
        return len(self.sample_files_by_index)

    def get(self, index):
        with gzip.open(self.sample_files_by_index[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample['data']
        constraint_features, (edge_indices, edge_features), variable_features = sample_observation

        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))

        candidate_choice = torch.from_numpy(np.array(sample_action, dtype=np.int32)).type(
            torch.LongTensor)  # action index relative to candidates
        candidate_scores = torch.FloatTensor(np.nan_to_num(sample_scores))

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, len(candidates), candidate_choice, candidate_scores, len(sample_scores))
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

        return graph


class TemporalGraphDataset(torch_geometric.data.Dataset):
    '''
    This dataset takes 32 consecutive iterations belonging to the same problem in a batch
    '''

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        sample_files.sort()
        indices = list(dict.fromkeys([t.split('.pkl')[0].split('_')[-2] for t in sample_files]))
        samples = {i: [] for i in indices}

        for t in sample_files:
            samples[t.split('.pkl')[0].split('_')[-2]].append(t)

        final_sample_indices = []
        for item in samples.keys():
            samples[item].sort()
            if len(samples[item]) >= 32:
                final_sample_indices.append(int(item))
        self.sample_files_by_index = dict.fromkeys([i for i in range(0, len(final_sample_indices))])

        for i, item in enumerate(final_sample_indices):
            self.sample_files_by_index[i] = samples['{:05d}'.format(item)][0:32]

    def len(self):
        return len(self.sample_files_by_index)

    def get(self, index):

        constraint_features_t, edge_indices_t, edge_features_t, variable_features_t, candidates_t, \
        candidate_choice_t, candidate_scores_t = [], [], [], [], [], [], []
        final_graph = []
        for item in self.sample_files_by_index[index]:
            with gzip.open(item, 'rb') as f:
                sample = pickle.load(f)

            sample_observation, sample_action, sample_action_set, sample_scores = sample['data']

            constraint_features, (edge_indices, edge_features), variable_features = sample_observation
            constraint_features = torch.FloatTensor(constraint_features)
            edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
            edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
            variable_features = torch.FloatTensor(variable_features)

            candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
            candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
            candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

            graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                      candidates, len(candidates), candidate_choice, candidate_scores)
            graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
            final_graph.append(graph)

        return Batch.from_data_list(final_graph, [], [])


class TemporalGraphDataset(torch_geometric.data.Dataset):
    '''
    This dataset diviedes the batch to blocks of 4 consecutive iterations belonging to the same problem
    '''

    def __init__(self, args, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.args = args
        sample_files.sort()

        indices = list(dict.fromkeys([t.split('.pkl')[0].split('_')[-2] for t in sample_files]))
        samples = {i: [] for i in indices}

        for t in sample_files:
            samples[t.split('.pkl')[0].split('_')[-2]].append(t)

        final_sample_indices = []
        for item in samples.keys():
            samples[item].sort()
            if len(samples[item]) >= args.GRU_temp_dim:
                final_sample_indices.append(int(item))
        self.sample_files_by_index = dict.fromkeys([i for i in range(0, len(final_sample_indices))])

        for i, item in enumerate(final_sample_indices):
            self.sample_files_by_index[i] = samples['{:05d}'.format(item)][0:32]

    def len(self):
        return len(self.sample_files_by_index)

    def get(self, index):

        constraint_features_t, edge_indices_t, edge_features_t, variable_features_t, candidates_t, \
        candidate_choice_t, candidate_scores_t = [], [], [], [], [], [], []

        result = [self.sample_files_by_index[index][i:i + self.args.GRU_temp_dim] for i in
                  range(len(self.sample_files_by_index[index]) - self.args.GRU_temp_dim + 1)]

        final_graph = []
        for i, list_ in enumerate(result):
            inter_graph = []
            for j, item in enumerate(list_):
                with gzip.open(item, 'rb') as f:
                    sample = pickle.load(f)

                sample_observation, sample_action, sample_action_set, sample_scores = sample['data']

                constraint_features, (edge_indices, edge_features), variable_features = sample_observation
                constraint_features = torch.FloatTensor(constraint_features)
                edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
                edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
                variable_features = torch.FloatTensor(variable_features)
                if self.args.problem == 'load_balancing':
                    min_ = max(0, 4500 - variable_features.shape[0])
                    variable_features = F.pad(variable_features, (0, 0, 0, min_))
                candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
                candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
                candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

                graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                          candidates, len(candidates), candidate_choice, candidate_scores)
                graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
                inter_graph.append(graph)
            final_graph.append(Batch.from_data_list(inter_graph, [], []))

        return final_graph


# Policies
class GCNNEvaluatePolicy:
    def __init__(self, args):
        self.rng = np.random.RandomState()

        # get parameters

        # set up policy
        self.device = args.device
        if args.pytorch_gat_II or args.pytorch_gat_I:
            print('Loading GAT NON TEMP POLICY ... ')
            self.policy = GATNonTempPolicy(args).eval().to(self.device)
        else:
            print('Loading GCNN POLICY ... ')
            self.policy = GNNPolicy(args).eval().to(self.device)
        msg = self.policy.load_state_dict(torch.load(args.cache_ckpt_path, map_location=args.device))
        print('---------------------------------------')
        print(f'msg: {msg} loaded from {args.cache_ckpt_path} to device: {args.device}', 'blue')
        print('---------------------------------------')

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        # mask variable features (no incumbent info)
        variable_features = observation.column_features
        constraint_features = torch.FloatTensor(observation.row_features).to(self.device)
        edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int32)).to(self.device)
        edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1)).to(self.device)
        variable_features = torch.FloatTensor(variable_features).to(self.device)
        action_set = torch.LongTensor(np.array(action_set, dtype=np.int32)).to(self.device)
        logits = self.policy(constraint_features, edge_index, edge_attr, variable_features)

        logits = logits[action_set]
        action_idx = logits.argmax().item()
        action = action_set[action_idx]
        return action



class GATGCNNEvaluatePolicy:
    def __init__(self, args):
        self.rng = np.random.RandomState()

        # get parameters
        # set up policy
        self.device = args.device
        self.policy = TGATPolicy(args).eval().to(self.device)
        msg = self.policy.load_state_dict(torch.load(args.cache_ckpt_path, map_location=args.device))
        print('---------------------------------------')
        print(f'msg: {msg} loaded from {args.cache_ckpt_path} to device: {args.device} Agent: {args.agent}', 'blue')
        print('---------------------------------------')

        self.constraint_features = []
        self.edge_index = []
        self.variable_features = []
        self.edge_attr = []
        self.action_set = []
        self.args = args

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, observation, varFeatures):
        with torch.no_grad():
            constraint_features = torch.from_numpy(observation.row_features.astype(np.float32)).to(self.device)
            edge_indices = torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(self.device)
            edge_features = torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(
                self.device)
            variable_features = torch.from_numpy(observation.column_features.astype(np.float32)).to(self.device)
            if self.args.problem == 'load_balancing':
                min_ = max(0, 4500 - variable_features.shape[0])
                variable_features = F.pad(variable_features, (0, 0, 0, min_))

            variable_features = self.policy.forward(constraint_features, edge_indices, edge_features, variable_features)
            varFeatures.append(variable_features)

            return varFeatures

    def branch(self, sample_action_set, observation, varFeatures):
        with torch.no_grad():
            constraint_features = torch.from_numpy(observation.row_features.astype(np.float32)).to(self.device)
            edge_indices = torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(self.device)
            edge_features = torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(
                self.device)
            variable_features = torch.from_numpy(observation.column_features.astype(np.float32)).to(self.device)
            if self.args.problem == 'load_balancing':
                min_ = max(0, 4500 - variable_features.shape[0])
                variable_features = F.pad(variable_features, (0, 0, 0, min_))

            variable_features = self.policy.forward(constraint_features, edge_indices, edge_features, variable_features)

            varFeatures.append(variable_features)
            variable_features = torch.stack(varFeatures)
            logits = self.policy.gru_fnc(variable_features)
            logits = logits.chunk(self.args.GRU_temp_dim)[-1]
            action = sample_action_set[logits[sample_action_set.astype(np.int64)].argmax()]
            varFeatures.pop(0)

            return action, varFeatures

