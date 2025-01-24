from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import sys
import argparse
import pathlib
import numpy as np
from constants import *
import builtins
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from utilities import *
from model.model_tgat import GNNPolicy as GATPolicy

import numpy as np

from environments import Branching as Environment
from rewards import TimeLimitDualIntegral as BoundIntegral
from utilities import BipartiteNodeData
from torch_geometric.data import Batch

from environments import Branching as Environment
from rewards import TimeLimitDualIntegral as BoundIntegral
import csv
import json
import ecole
import sys

if os.getcwd().split('/')[2] == 'mehdi':
    from PColor import ColorPrint
    print = ColorPrint().print


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='setcover', help='MILP instance type',
                        choices=['setcover','cauctions', 'load_balancing', 'anonymous', 'indset', 'facilities'])
    parser.add_argument('--samples_path', type=str, default='data/samples/')
    parser.add_argument('--logs_dir', type=str, default='logs/')
    parser.add_argument('-s', '--seed', help='Random generator seed.', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pretrain_batch_size', type=int, default=1)
    parser.add_argument('--valid_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--entropy_bonus', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)


    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--distributed', type=bool_flag, default=True)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--version', type=str)


    parser.add_argument('--data_url')  # Just so ROMA doesn't complain
    parser.add_argument('--init_method')
    parser.add_argument('--train_url')
    parser.add_argument('--dir_path', default='.')

    # Bipartitde graph parameters
    parser.add_argument('--emb_size', default=32, type=int)

    # GAT parameters
    parser.add_argument('--num_of_layers', default=1, type=int)
    parser.add_argument('--num_heads_per_layer', nargs="+", type=int, default=[2])
    parser.add_argument('--num_features_per_layer')
    parser.add_argument('--add_skip_connection', type=bool_flag, default=True)
    parser.add_argument('--bias', type=bool_flag, default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument('--gat_config')
    parser.add_argument('--pytorch_gat', help='use teh pytorchgat', action = 'store_true')
    parser.add_argument('--pytorch_gat_I', help='use teh pytorchgat', action = 'store_true')
    parser.add_argument('--pytorch_gat_II', help='use teh pytorchgat', action = 'store_true')
    parser.add_argument('--share_weights', help='share attention weights', action = 'store_true')
    parser.add_argument('--fill_value', type=str, default='mean', help='attention layer aggregation mechanism',
                        choices=['mean','max', 'min', 'add', 'mul'])
    parser.add_argument('--concat', help='concat attention', action = 'store_true')
    parser.add_argument('--elu', help='elu non linearity', action='store_true')

    #GRU parameters
    parser.add_argument('--GRU_temp_dim', type=int, default=4)
    parser.add_argument('--GRU_hidden_dim', type=int,default=32)
    parser.add_argument('--GRU_input_dim', type=int)
    parser.add_argument('--GRU_num_layers', type=int, default=1)
    parser.add_argument('--GRU_bidirectional', type=bool_flag, default=True)



    # evaluation parameters

    parser.add_argument(
        '-t', '--timelimit',
        help='Episode time limit (in seconds).',
        default=argparse.SUPPRESS,
        type=float,
    )
    parser.add_argument(
        '-d', '--debug',
        help='Print debug traces.',
        action='store_true',
    )
    parser.add_argument(
        '-f', '--folder',
        help='Instance folder to evaluate.',
        default="valid",
        type=str,
        choices=("valid", "test"),
    )

    args, unparsed = parser.parse_known_args()
    args.num_features_per_layer = [args.emb_size for item in range(len(args.num_heads_per_layer) + 1)]

    args.pytorch_gat = args.pytorch_gat_I or args.pytorch_gat_II

    if args.concat:
        args.GRU_input_dim = args.emb_size * args.num_heads_per_layer[0]
        args.GRU_hidden_dim = args.emb_size
    else:
        args.GRU_input_dim = args.emb_size
        args.GRU_hidden_dim = args.emb_size


    args.gat_config = {
        "num_of_layers": args.num_of_layers,
        "num_heads_per_layer": args.num_heads_per_layer,
        "num_features_per_layer": args.num_features_per_layer,
        "add_skip_connection": args.add_skip_connection,
        "bias": args.bias,
        "dropout": args.dropout,
        'entropy_bonus': args.entropy_bonus,
        'top_k': args.top_k,
        'fill_value':args.fill_value,
        'share_weights':args.share_weights,
        'concat':args.concat,
        'elu':args.elu
    }

    # multi processing
    args.distributed = (torch.cuda.device_count() > 1) and (args.distributed)
    if args.distributed:
        args.gpu = 'cuda:%d' % args.local_rank

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method)

    args.world_size = torch.distributed.get_world_size()

    #suppress printing if not master
    if args.gpu != 'cuda:0':
        def print_pass(*args):
            pass

        builtins.print = print_pass

    args.rank = torch.distributed.get_rank()
    args.batch_size = args.batch_size
    args.pretrain_batch_size = args.pretrain_batch_size
    args.valid_batch_size = args.valid_batch_size
    args.logs_dir = os.path.join(args.logs_dir, f'v{args.version}', args.problem)
    args.saved_file_path = os.path.join(args.dir_path, args.logs_dir, 'files/model_tgat.py')

    problem_folders = {
        'setcover': 'setcover_temporal_32/500r_1000c_0.05d',
        'cauctions': 'cauctions_temporal_32/100_500',
        'facilities': 'facilities_temporal_32/100_100_5',
        'indset': 'indset_temporal_32/500_4',
        'mknapsack': 'mknapsack_temporal_32/100_6',
        'item_placement': 'item_placement_temporal_32/1_item_placement',
        'load_balancing': 'load_balancing_temporal_4/2_load_balancing',
        'anonymous': 'anonymous_temporal_32/3_anonymous'
    }

    if not os.path.isdir(args.logs_dir):
        os.makedirs(args.logs_dir,exist_ok=True)
    if not os.path.isdir(os.path.join(args.logs_dir,'ckpt')):
        os.makedirs(os.path.join(args.logs_dir,'ckpt'),exist_ok=True)
    if not os.path.isdir(os.path.join(args.logs_dir, 'files')):
        os.makedirs(os.path.join(args.logs_dir, 'files'),exist_ok=True)
    if not os.path.isdir(os.path.join(args.logs_dir, 'results')):
        os.makedirs(os.path.join(args.logs_dir, 'results'),exist_ok=True)

    samples_path = args.samples_path

    args.samples_root_path = os.path.realpath(samples_path)
    args.samples_path = os.path.join(args.samples_root_path, problem_folders[args.problem])

    print('=====================================================================')
    print(f'torch: {torch.__version__}  geometric: {torch_geometric.__version__}')
    print('=====================================================================')


    # Gat setup


    args.device = args.gpu#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # randomization setup
    args.rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    arg = vars(args)
    for item in arg:
        print('{:.<30s}{:<50s}'.format(str(item), str(arg[item])))

    policy = GATPolicy(args).cuda()

    if args.ckpt_path:
        # load weight from pretrained-model
        policy.load_state_dict(torch.load(args.ckpt_path))
        print("load weight from {}".format(args.ckpt_path))

    # get sample directory
    if not args.samples_path.endswith('/'):
        args.samples_path += '/'

    print(f'Sample Path: {args.samples_path}')
    train_files = glob.glob(args.samples_path + 'train/sample_*.pkl')
    valid_files = glob.glob(args.samples_path + 'valid/sample_*.pkl')


    print("Number of train files: {}".format(len(train_files)))
    print("Number of validation files: {}".format(len(valid_files)))
    print(f'number of model parameters: {count_parameters(policy)}','red')


    train_gnn(train_files, valid_files, policy, args)


def train_gnn(train_files, valid_files, policy, args):

    # data setup
    train_data = TemporalGraphDataset(args, train_files)
    valid_data = TemporalGraphDataset(args, valid_files)
    pretrain_indices = [i for i in range(0, len(train_data), 10)]



    sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, sampler=sampler,drop_last=True)
    sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, sampler=sampler,drop_last=True)
    sampler = torch.utils.data.SubsetRandomSampler(pretrain_indices)
    pretrain_loader = DataLoader(train_data, batch_size=args.pretrain_batch_size, shuffle=False, sampler=sampler)

    print("Number of train iterations: {} {}".format(len(train_data), len(train_loader)))
    print("Number of validation iterations: {} {}".format(len(valid_data), len(valid_loader)))

    best_acc = -1
    best_epoch = -1

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = Scheduler(optimizer, mode='min', patience=10, factor=0.2, verbose=True)
    policy = DDP(policy, device_ids=[args.local_rank])
    args.cache_ckpt_path = ''
    ckpt_file = ''
    for epoch in range(args.max_epochs):
        if args.local_rank == 0:
            print('-------------------------------------------')
        train_loader.sampler.set_epoch(epoch)
        # print("distributed: ", distributed)
        if args.debug:
           epoch = 1
        if epoch == 0:
            if args.local_rank == 0:
                print("Pre-training the pre-norm layers...")
            n = pretrain(policy, pretrain_loader, args)
            if args.local_rank == 0:
                print(f"PRETRAINED {n} LAYERS")
        else:
            train_loss, train_kacc = process(policy, train_loader, args, optimizer)

            if args.local_rank == 0:
                print(f"Epoch: [{epoch}] TRAIN LOSS: {train_loss:0.3f} " + "".join(
                    [f" acc@{k}: {acc:0.3f}" for k, acc in zip(args.top_k, train_kacc)]))

        # validate
        valid_loss, valid_acc = process(policy, valid_loader, args, None)
        if args.local_rank == 0:
            print(f"Epoch: [{epoch}] VALID LOSS: {valid_loss:0.3f} " + "".join(
                [f" acc@{k}: {acc:0.3f}" for k, acc in zip(args.gat_config['top_k'], valid_acc)]))

        # scheduler.step()
        scheduler.step(valid_loss)
        if scheduler.num_bad_epochs == 10:
            print(f"  10 epochs without improvement, decreasing learning rate")
        elif scheduler.num_bad_epochs == 20:
            print(f"  20 epochs without improvement, early stopping")
            break

        if args.local_rank == 0:
            if valid_acc[0] > best_acc:
                best_acc = valid_acc[0]
                best_epoch = epoch
                ckpt_file = f'best_trained_params_{args.problem}_v{args.version}.pkl'
                args.cache_ckpt_path = os.path.join(args.logs_dir,'ckpt', ckpt_file)
                torch.save(policy.module.state_dict(), args.cache_ckpt_path)
                print("New best epoch: {}".format(best_epoch))

                print("Best epoch: {}: {} ".format(best_epoch, best_acc))


def pretrain(policy, pretrain_loader, args):
    """
    Pre-trains all PreNorm layers in the model.

    Parameters
    ----------
    policy : torch.nn.Module
        Model to pre-train.
    pretrain_loader : torch_geometric.loader.DataLoader
        Pre-loaded dataset of pre-training samples.

    Returns
    -------
    i : int
        Number of pre-trained layers.
    """
    policy.module.pre_train_init()

    i = 0
    N = len(pretrain_loader)
    while True:
        for cnt, batch_ in enumerate(pretrain_loader):
            batch = batch_[0].to(args.device)
            #graph = graph.to(args.device)

            if batch is not None:
                if args.distributed:
                    if not policy.module.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr,
                                                   batch.variable_features):
                        break
                else:
                    if not policy.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr,
                                            batch.variable_features):
                        break
        if args.distributed:
            if policy.module.pre_train_next() is None:
                break
        else:
            if policy.pre_train_next() is None:
                break
        i += 1
    return i


def process(policy, data_loader, args, optimizer=None):
    print('Processing ...')
    mean_loss = 0
    top_k = args.gat_config['top_k']
    mean_kacc = np.zeros(len(top_k))
    mean_entropy = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for i, batch_ in enumerate(data_loader):
            loss = 0
            for batch in batch_:
                batch = batch.to(args.device)
                gru_input = policy.module.forward(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                gru_input   =gru_input.view(args.GRU_temp_dim, -1 , args.GRU_input_dim)
                logits = policy.module.gru_fnc(gru_input)
                logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
                cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
                entropy = (-F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                loss += (cross_entropy_loss - args.entropy_bonus * entropy)

                true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
                true_bestscore = true_scores.max(dim=-1, keepdims=True).values

                kacc = []
                for k in top_k:
                    if logits.size()[-1] < k:
                        kacc.append(1.0)
                        continue
                    pred_top_k = logits.topk(k).indices
                    pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
                    accuracy = (pred_top_k_true_scores == true_bestscore).any(dim=-1).float().mean().item()
                    kacc.append(accuracy)
                kacc = np.asarray(kacc)
                mean_kacc += kacc * batch.num_graphs

                if args.distributed:
                    reduced_cross_entropy_loss = reduce_tensor(cross_entropy_loss, args.world_size)
                    reduced_entropy = reduce_tensor(entropy, args.world_size)
                else:
                    reduced_cross_entropy = cross_entropy_loss
                    reduced_entropy = entropy

                mean_loss += reduced_cross_entropy_loss.item() * batch.num_graphs
                mean_entropy += reduced_entropy.item() * batch.num_graphs

                n_samples_processed += batch.num_graphs

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    mean_entropy /= n_samples_processed

    return mean_loss, mean_kacc



if __name__ == "__main__":

    main()

