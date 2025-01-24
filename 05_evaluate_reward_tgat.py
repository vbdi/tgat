import argparse
import csv
import json
import pathlib
import ecole as ec
import numpy as np
import torch
import pandas as pd
import time
from constants import *
from environments import Branching as Environment
from rewards import TimeLimitDualIntegral as BoundIntegral
from utilities import General, Random
from utilities import ObservationFunction, Psebservation, FSBObservation, GCNNEvaluatePolicy, GATGCNNEvaluatePolicy, ExploreThenStrongBranch
import os
from datetime import datetime
import copy

epsilon = 1e-6
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


def main(args):
 
    args.logs_dir = os.path.join(args.logs_dir, f'v{args.version}', args.problem)
    #######################################
    # Create obs and cache folder
    if not os.path.isdir(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)
    if not os.path.isdir(os.path.join(args.logs_dir, 'ckpt')):
        os.makedirs(os.path.join(args.logs_dir, 'ckpt'), exist_ok=True)
    if not os.path.isdir(os.path.join(args.logs_dir, 'files')):
        os.makedirs(os.path.join(args.logs_dir, 'files'), exist_ok=True)
    if not os.path.isdir(os.path.join(args.logs_dir, 'results')):
        os.makedirs(os.path.join(args.logs_dir, 'results'), exist_ok=True)

    #######################################

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    samples_path = args.samples_path

    args.samples_root_path = os.path.realpath(samples_path)

    arg = vars(args)
    for item in arg:
        print('{:.<30s}{:<50s}'.format(str(item), str(arg[item])))

    args.instances = []
    if args.problem == 'setcover':
        args.instances += [
            {'type': 'small',
             'path': f"{args.samples_root_path}/setcover/instances/transfer_500r_1000c_0.05d/instance_{i + 1}.lp"}
            for i in range(20)]
        args.instances += [
            {'type': 'medium',
             'path': f"{args.samples_root_path}/setcover/instances/transfer_1000r_1000c_0.05d/instance_{i + 1}.lp"}
            for i in range(20)]
        args.instances += [
            {'type': 'big', 'path': f"{args.samples_root_path}/setcover/instances/transfer_2000r_1000c_0.05d/instance_{i + 1}.lp"}
            for i in range(20)]

    elif args.problem == 'cauctions':
        args.instances += [
            {'type': 'small',
             'path': f"{args.samples_root_path}/cauctions/instances/transfer_100_500/instance_{i + 1}.lp"}
            for i in
            range(20)]
        args.instances += [
            {'type': 'medium',
             'path': f"{args.samples_root_path}/cauctions/instances/transfer_200_1000/instance_{i + 1}.lp"}
            for i
            in range(20)]
        args.instances += [
            {'type': 'big',
             'path': f"{args.samples_root_path}/cauctions/instances/transfer_300_1500/instance_{i + 1}.lp"}
            for i in
            range(20)]

    elif args.problem == 'facilities':
        args.instances += [
            {'type': 'small',
             'path': f"{args.samples_root_path}/facilities/instances/transfer_100_100_5/instance_{i + 1}.lp"}
            for i
            in range(20)]
        args.instances += [{'type': 'medium',
                            'path': f"{args.samples_root_path}/facilities/instances/transfer_200_100_5/instance_{i + 1}.lp"}
                           for i in range(20)]
        args.instances += [{'type': 'big',
                            'path': f"{args.samples_root_path}/facilities/instances/transfer_400_100_5/instance_{i + 1}.lp"}
                           for
                           i in range(20)]

    elif args.problem == 'indset':
        args.instances += [{'type': 'small',
                            'path': f"{args.samples_root_path}/indset/instances/transfer_500_4/instance_{i + 1}.lp"}
                           for i in
                           range(20)]
        args.instances += [{'type': 'medium',
                            'path': f"{args.samples_root_path}/indset/instances/transfer_1000_4/instance_{i + 1}.lp"}
                           for i
                           in range(20)]
        args.instances += [{'type': 'big',
                            'path': f"{args.samples_root_path}/indset/instances/transfer_1500_4/instance_{i + 1}.lp"}
                           for i in
                           range(20)]

    elif args.problem == 'item_placement':
        args.instances += [{'type': 'mixed',
                            'path': f"{args.samples_root_path}/item_placement/instances/1_item_placement"
                                    f"/test/item_placement_{i + 1}.mps.gz"} for i in range(9999, 10099)]

    elif args.problem == 'load_balancing':
        args.instances += [{'type': 'mixed',
                            'path': f"{args.samples_root_path}/load_balancing_temporal_4/instances/2_load_balancing"
                                    f"/test/load_balancing_{i + 1}.mps.gz"} for i in range(9999, 10099)]
    elif args.problem == 'anonymous':
        args.instances += [{'type': 'mixed',
                            'path': f"{args.samples_root_path}/anonymous/instances/3_anonymous"
                                    f"/test/anonymous_{i + 1}.mps.gz"} for i in range(118, 138)]

    else:
        NotImplementedError

    ckpt_file = f'best_trained_params_{args.problem}_v{args.version}.pkl'
    args.cache_ckpt_path = os.path.join(args.dir_path, args.logs_dir, 'ckpt', ckpt_file)

    results_fieldnames = ['instance', 'expert_prob', 'seed', 'objective_offset',
                          'cumulated_reward', 'stime', 'nnodes', 'nlps', 'gap', 'status']

    outcome = {'agent': [], 'expert_prob': [], 'time_limit': [], 'reward_time': [],
               'nnodes': [], 'iter': [], 'stime': [], 'gap': []}

    import sys
    sys.path.insert(1, str(pathlib.Path.cwd()))

    # set up the proper agent, environment and goal for the task
    expert_probability = [0]
    args.timelimit = [3600, 2400, 1200,900, 480, 240, 120, 60]

    score_function = [ec.observation.Pseudocosts(), ec.observation.StrongBranchingScores()]
    integral_function = BoundIntegral()
    iter = 0
    for expert_prob in expert_probability:
        for time_limit in args.timelimit:
            agent = args.agent
            args.expert_prob = expert_prob
            processing_time = []
            if args.agent == "Temporal":
                policy = GATGCNNEvaluatePolicy(args)
                observation_function_prep = {"node_observation": ec.observation.NodeBipartite(),
                                             "scores": score_function[expert_prob]}
                observation_function = {"node_observation": ec.observation.NodeBipartite(), "scores":None}
            else:
                NotImplementedError
            # override from command-line argument if provided
            dd = datetime.today().strftime('%Y-%m-%d')
            dt = datetime.today().strftime('%H-%M-%S')

            cache_results_file = f"{args.logs_dir}/results/{args.problem}_{time_limit}_{args.agent}_{args.expert_prob}_{dd}_{dt}.csv"
            obs_results_file = f"{args.dir_path}/{args.logs_dir}/results/{args.problem}_{time_limit}_{args.agent}_{args.expert_prob}_{dd}_{dt}.csv"

            with open(cache_results_file, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
                writer.writeheader()

            # evaluation loop
            for seed, instance in enumerate(args.instances):
                seed += args.seed
                try:
                    # seed both the agent and the environment (deterministic behavior)
                    env_prep = Environment(
                        time_limit=time_limit,
                        observation_function=observation_function_prep,
                        reward_function=-integral_function  # negated integral (minimization)
                    )

                    policy.seed(seed)
                    env_prep.seed(seed)
                    # read the instance's initial primal and dual bounds from JSON file
                    if args.problem in ['item_placement', 'load_balancing', 'anonymous']:
                        json_file = instance['path'].split('.mps.gz')[0] + '.json'
                        with open(json_file) as f:
                            instance_info = json.load(f)
                        # set up the reward function parameters for that instance
                        initial_primal_bound = instance_info["primal_bound"]
                        initial_dual_bound = instance_info["dual_bound"]
                    else:
                        initial_primal_bound = None
                        initial_dual_bound = None
                    objective_offset = 0
                    cumulated_reward = 0  # discard initial reward

                    integral_function.set_parameters(
                        initial_primal_bound=initial_primal_bound,
                        initial_dual_bound=initial_dual_bound,
                        objective_offset=objective_offset)

                    print()
                    print(f"  instance {instance['path']}")
                    print(f"  seed: {seed} / {len(args.instances)}")
                    print(f"  initial primal bound: {initial_primal_bound}")
                    print(f"  initial dual bound: {initial_dual_bound}")
                    print(f"  objective offset: {objective_offset}")
                    print(f"  time_limit: {time_limit}")
                    print(f"  agent: {agent}")
                    print(f"  expert_prob: {expert_prob}")
                    print(f"  observation_function_prep: {observation_function_prep}")
                    print(f"  observation_function: {observation_function}")
                    observation, sample_action_set, _, done, _ = env_prep.reset(instance['path'],objective_limit=initial_primal_bound)
                    iter = 0
                    gruInput = []
                    # loop over the environment
                    while iter < args.GRU_temp_dim - 1:
                        scores = observation["scores"]
                        scores = np.nan_to_num(scores, 0.0)
                        gruInput = policy(observation['node_observation'], gruInput)
                        action = sample_action_set[scores[sample_action_set].argmax()]
                        observation, sample_action_set, reward, _, _ = env_prep.step(action)
                        cumulated_reward += reward
                        iter += 1
                    env_prep.observation_function =  ec.data.parse(observation_function, ec.observation.Nothing)
                    while not done:
                        action, gruInput = policy.branch(sample_action_set, observation['node_observation'],
                                                         gruInput)
                        observation, sample_action_set, reward, done, info = env_prep.step(action)
                        cumulated_reward += reward
                        iter += 1

                    scip_model = env_prep.model.as_pyscipopt()
                    stime = scip_model.getSolvingTime()
                    nnodes = scip_model.getNNodes()
                    nlps = scip_model.getNLPs()
                    gap = scip_model.getGap()
                    status = scip_model.getStatus()
                    processing_time.append(stime)

                    print(f"  cumulated reward (to be maximized): {cumulated_reward:.2f}\t "
                           f" stime: {stime:.2f} nodes_solved: {nnodes} gap: {gap:.2e} status: {status}")
                    # save instance results
                    with open(cache_results_file, mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
                        writer.writerow({
                            'instance': instance['path'],
                            'seed': seed,
                            'expert_prob': expert_prob,
                            'objective_offset': objective_offset,
                            'cumulated_reward': cumulated_reward,
                            'stime': stime,
                            'nnodes': nnodes,
                            'nlps': nlps,
                            'gap': gap,
                            'status': status})

                except Exception as e:
                    print(f'Exception: {e} number of iterations: {iter}/{args.GRU_temp_dim}', 'red')
                    scip_model = env_prep.model.as_pyscipopt()
                    stime = scip_model.getSolvingTime()
                    nnodes = scip_model.getNNodes()
                    nlps = scip_model.getNLPs()
                    gap = scip_model.getGap()
                    status = scip_model.getStatus()

                    total_time = scip_model.getParam("limits/time")

                    print(f"  cumulated reward (to be maximized): {cumulated_reward:.2f} "
                           f"stime: {stime:.2f} nodes_solved: {nnodes} gap: {gap:.2e} status: {status} ")
                    processing_time.append(stime)

                    # save instance results
                    with open(cache_results_file, mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
                        writer.writerow({
                            'instance': instance['path'],
                            'seed': seed,
                            'expert_prob': expert_prob,
                            # 'initial_primal_bound': initial_primal_bound,
                            # 'initial_dual_bound': initial_dual_bound,
                            'objective_offset': objective_offset,
                            'cumulated_reward': cumulated_reward,
                            'stime': stime,
                            'nnodes': nnodes,
                            'nlps': nlps,
                            'gap': gap,
                            'status': 'error'})

            avg_initial_primal_bound, avg_initial_dual_bound, avg_objective_offset, avg_cumulated_reward, \
            cum_nnodes, avg_stime, avg_gap = get_metrics(args, cache_results_file, time_limit=time_limit)

            outcome['agent'].append(agent)
            outcome['expert_prob'].append(expert_prob)
            outcome['time_limit'].append(time_limit)
            outcome['reward_time'].append(avg_cumulated_reward)
            outcome['nnodes'].append(cum_nnodes)
            outcome['iter'].append(iter - 1)
            outcome['stime'].append(avg_stime)
            outcome['gap'].append(avg_gap)

            print('-----------------------------------------SUMMARY------------------------------------------------')
            LEN = len(outcome['reward_time'])
            KEYS = outcome.keys()
            for k0 in range(LEN):
                for k1 in KEYS:
                    if isinstance(outcome[k1][k0], str):
                        print('{:<2}:  {}'.format(k1, outcome[k1][k0][0:3]), end='  ')
                    elif k1 == 'gap':
                        print('{:<2}:  {:.2e}'.format(k1, outcome[k1][k0]), end='  ')
                    elif k1 in ['nnodes', 'iter', 'time_limit']:
                        print('{:<2}:  {:d}'.format(k1, outcome[k1][k0]), end='    ')
                    else:
                        print('{:<2}:  {:.2f}'.format(k1, outcome[k1][k0]), end='    ')
                print('\n')

            print(
                f'--------------------------------------------{args.agent}:{time_limit}--------------------------------------------------')
            print(processing_time)
            print('------------------------------------------------------------------------------------------------')

            if args.use_roma:
                print('===========================================================')
                print(f'copying csv files from {cache_results_file} to {obs_results_file}')
                print('===========================================================')
                try:
                    if not mox.file.is_directory(f'{args.dir_path}/{args.logs_dir}/results/'):
                        mox.file.make_dirs(f'{args.dir_path}/{args.logs_dir}/results/')

                    mox.file.copy(cache_results_file, obs_results_file)

                except Exception as E:
                    print(f"error: {E}")
                    print("Could not copy to S3")


def get_metrics(args, results_csv_path, time_limit=1 * 60):
    print('Printing Evaluation Metrics...')
    df = pd.read_csv(results_csv_path)
    avg_initial_primal_bound = None
    avg_initial_dual_bound = None

    # avg_initial_primal_bound = df["initial_primal_bound"].mean()
    # avg_initial_dual_bound = df["initial_dual_bound"].mean()
    avg_objective_offset = df["objective_offset"].mean()
    avg_cumulated_reward = df["cumulated_reward"].mean()

    avg_stime = df['stime'].mean()
    avg_gap = df['gap'].mean()
    cum_nnodes = df['nnodes'].sum()

    # print("avg_initial_primal_bound: {}".format(avg_initial_primal_bound))
    # print("avg_initial_dual_bound: {}".format(avg_initial_dual_bound))
    t = df['instance']
    r = df['cumulated_reward']
    nnodes = df['nnodes']
    tt = df['stime']

    if args.problem == 'cauctions':
        s_index_G = [i for i, t in t.items() if '100_500' in t]
        m_index_G = [i for i, t in t.items() if '200_1000' in t]
        b_index_G = [i for i, t in t.items() if '300_1500' in t]
    elif args.problem == 'setcover':
        s_index_G = [i for i, t in t.items() if '500r_1000c' in t]
        m_index_G = [i for i, t in t.items() if '1000r_1000c' in t]
        b_index_G = [i for i, t in t.items() if '2000r_1000c' in t]
    elif args.problem == 'indset':
        s_index_G = [i for i, t in t.items() if '500_4' in t and not '1500_4' in t]
        m_index_G = [i for i, t in t.items() if '1000_4' in t]
        b_index_G = [i for i, t in t.items() if '1500_4' in t]
    elif args.problem == 'facilities':
        s_index_G = [i for i, t in t.items() if '100_100_5' in t]
        m_index_G = [i for i, t in t.items() if '200_100_5' in t]
        b_index_G = [i for i, t in t.items() if '400_100_5' in t]
    else:
        s_index_G = [i for i, t in t.items()]
        m_index_G = [0 for i, t in t.items()]
        b_index_G = [0 for i, t in t.items()]

    s_reward_GCN = r[s_index_G].mean()
    m_reward_GCN = r[m_index_G].mean()
    b_reward_GCN = r[b_index_G].mean()

    s_nn_GCN = nnodes[s_index_G].mean()
    m_nn_GCN = nnodes[m_index_G].mean()
    b_nn_GCN = nnodes[b_index_G].mean()

    s_time_GCN_a = tt[s_index_G].mean()
    m_time_GCN_a = tt[m_index_G].mean()
    b_time_GCN_a = tt[b_index_G].mean()

    if len(s_index_G) > 0:
        s_time_GCN = (tt[s_index_G] + 1).prod() ** (1.0 / len(s_index_G))
    else:
        s_time_GCN = 0

    if len(m_index_G) > 0:
        m_time_GCN = (tt[m_index_G] + 1).prod() ** (1.0 / len(m_index_G))
    else:
        m_time_GCN = 0
    if len(b_index_G) > 0:
        b_time_GCN = (tt[b_index_G] + 1).prod() ** (1.0 / len(b_index_G))
    else:
        b_time_GCN = 0

    print(f'------------------------AVG CUM REWARD Temporal-------------------------\n')
    print(f'Time_reward:  \t{s_reward_GCN:.2f}\t{m_reward_GCN:.2f}\t{b_reward_GCN:.2f}')
    print(f'nnode:        \t{s_nn_GCN:.2f}\t{m_nn_GCN:.2f}\t{b_nn_GCN:.2f}')
    print(f'geometric time:         \t{s_time_GCN:.2f}\t{m_time_GCN:.2f}\t{b_time_GCN:.2f}')
    print(f'arethmatic time:         \t{s_time_GCN_a:.2f}\t{m_time_GCN_a:.2f}\t{b_time_GCN_a:.2f}')

    return avg_initial_primal_bound, avg_initial_dual_bound, avg_objective_offset, avg_cumulated_reward, cum_nnodes, avg_stime, avg_gap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='setcover', help='MILP instance type',
                        choices=['setcover', 'cauctions', 'item_placement', 'load_balancing', 'anonymous', 'indset',
                                 'facilities'])
    parser.add_argument('--samples_path', type=str, default='data/samples/')
    parser.add_argument('--logs_dir', type=str, default='logs/')
    parser.add_argument('-s', '--seed', help='Random generator seed.', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pretrain_batch_size', type=int, default=1)
    parser.add_argument('--valid_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--entropy_bonus', type=float, default=0.0)

    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--version', type=str)
    parser.add_argument('--use_roma', help='use roma', action='store_true')

    # ROMA related args
    parser.add_argument('--data_url')  # Just so ROMA doesn't complain
    parser.add_argument('--init_method')
    parser.add_argument('--train_url')
    parser.add_argument('--dir_path', default='.')

    # Bipartitde graph parameters
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
    parser.add_argument('--remove_gat', help='remove_gat', action='store_true')
    parser.add_argument('--gat_first', help='remove_gat', action='store_true')
    parser.add_argument('--gat_interweave', help='interweave_gat', action='store_true')
    parser.add_argument('--remove_mpn', help='remove message passing network', action='store_true')
    parser.add_argument('--pytorch_gat', help='use teh pytorchgat', action='store_true')
    parser.add_argument('--pytorch_gat_I', help='use teh pytorchgat', action='store_true')
    parser.add_argument('--pytorch_gat_II', help='use teh pytorchgat', action='store_true')
    parser.add_argument('--share_weights', help='share attention weights', action = 'store_true')
    parser.add_argument('--fill_value', type=str, default='mean', help='attention layer aggregation mechanism',
                        choices=['mean','max', 'min', 'add', 'mul'])
    parser.add_argument('--concat', help='concat attention', action = 'store_true')
    parser.add_argument('--elu', help='elu non linearity', action='store_true')


    # GRU parameters
    parser.add_argument('--GRU_temp_dim', type=int, default=4)
    parser.add_argument('--GRU_hidden_dim', type=int, default=32)
    parser.add_argument('--GRU_input_dim', type=int)
    parser.add_argument('--GRU_num_layers', type=int, default=1)
    parser.add_argument('--GRU_bidirectional', type=bool_flag, default=True)
    parser.add_argument('--expert_prob', type=float, default=1.0)

    # evaluation parameters

    parser.add_argument('--timelimit', type=int, nargs="+", default=[900])

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
    parser.add_argument(
        '-a', '--agent',
        help='What agent you are evaluating.',
        default="Temporal",
        type=str,
        choices=("random", "Temporal", "GCNN", "FSB"),
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
        # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_of_layers": args.num_of_layers,
        # PPI has got 42% of nodes with all 0 features - that's why 3 layers are useful
        "num_heads_per_layer": args.num_heads_per_layer,
        # other values may give even better results from the reported ones
        "num_features_per_layer": args.num_features_per_layer,  # 64 would also give ~0.975 uF1!
        "add_skip_connection": args.add_skip_connection,
        # skip connection is very important! (keep it otherwise micro-F1 is almost 0)
        "bias": args.bias,  # bias doesn't matter that much
        "dropout": args.dropout,  # dropout hurts the performance (best to keep it at 0)
        "layer_type": LayerType.IMP3,  # the only implementation that supports the inductive setting
        'entropy_bonus': args.entropy_bonus,
        'top_k': args.top_k,
        'fill_value':args.fill_value,
        'share_weights':args.share_weights,
        'concat':args.concat,
        'elu':args.elu
    }

    main(args)