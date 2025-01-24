import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle
from utilities import *
import torch
import moxing as mox
import ecole
import pyscipopt
import pathlib
from constants import *
from torch_geometric.data import Batch
import pandas as pd

if os.getcwd().split('/')[2] == 'mehdi':
    from PColor import ColorPrint
    print = ColorPrint().print
from environments import DefaultInformationFunction


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
                        choices=['setcover','cauctions', 'item_placement', 'load_balancing', 'anonymous', 'indset', 'facilities'])
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
    parser.add_argument('--gpu', default=None, type=int,
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

    # GRU parameters
    parser.add_argument('--GRU_temp_dim', type=int, default=4)
    parser.add_argument('--GRU_hidden_dim', type=int, default=32)
    parser.add_argument('--GRU_input_dim', type=int)
    parser.add_argument('--GRU_num_layers', type=int, default=1)
    parser.add_argument('--GRU_bidirectional', type=bool_flag, default=True)

    # evaluation parameters

    parser.add_argument(
        '-t', '--timelimit',
        help='Episode time limit (in seconds).',
        default=60,
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
    parser.add_argument(
        '-a', '--agent',
        help='What agent you are evaluating.',
        default="Temporal",
        type=str,
        choices=("random", "Temporal", "GCNN", "FSB"),
    )

    args, unparsed = parser.parse_known_args()

    args.num_features_per_layer = [args.emb_size for item in range(len(args.num_heads_per_layer) + 1)]
    args.GRU_input_dim = args.emb_size

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
        'top_k': args.top_k, }


    #######################################
    # ROMA: Add these for ROMA support. Default is set to run local.
    #######################################
    # ROMA: Add these for ROMA support. Default is set to run local.
    if args.use_roma:
        import sys
        mox.file.shift('os', 'mox')
        args.this_file_path = os.path.dirname(os.path.realpath(__file__))
        code_base_dirpath = os.path.abspath(os.path.join(args.this_file_path, os.pardir))
        os.environ['TORCH_HOME'] = os.path.join(code_base_dirpath, '.torch_home')
        sys.path.append(code_base_dirpath)
        obs_root = os.path.join(args.dir_path, args.logs_dir)
        if not mox.file.is_directory(os.path.join(obs_root, f'v{args.version}', args.problem, 'results')):
            mox.file.make_dirs(os.path.join(obs_root, f'v{args.version}', args.problem, 'results'))
        if not mox.file.is_directory(os.path.join(obs_root, f'v{args.version}', args.problem, 'files')):
            mox.file.make_dirs(os.path.join(obs_root, f'v{args.version}', args.problem, 'files'))

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

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.use_roma:
        # Copy data from S3 to local of ROMA
        # dataset_root = 's3://vanbdai-share-cn1/Amin/datasets/ml4co/train_samples/training_anonymous_1.0_900.zip'
        dataset_root = args.samples_path
        cache_dirpath = '/cache/ml4co/'
        dst_url = os.path.join(cache_dirpath, "data_dir")
        dataset_zipfile = os.path.join(dst_url, os.path.basename(dataset_root))  # /cache/ml4co/data_dir/xxx.zip
        # print("Copying data from S3 to local of ROMA ...")
        print("Destination url: {}".format(dataset_zipfile))
        # Copying & unzipping is commented out from here as a separate script does that on ROMA on a single process
        # mox.file.copy(dataset_root, dataset_zipfile)
        print("File exists: ", os.path.exists(dataset_zipfile))
        # print("Copying done.")
        # print("Unzipping the dataset... {}".format(dataset_zipfile))
        # import zipfile
        # with zipfile.ZipFile(dataset_zipfile, 'r') as zip_ref:
        #     zip_ref.extractall(dst_url)
        # print("Unzipping the dataset... done")
        dataset_root = dst_url = dataset_zipfile
        print("dataset_root: {}".format(dataset_root))
        samples_path = dataset_root
    else:
        samples_path = args.samples_path

    args.samples_root_path = os.path.realpath(samples_path)
    arg = vars(args)
    for item in arg:
        print('{:.<30s}{:<50s}'.format(str(item), str(arg[item])))

    args.instances = []
    if args.problem == 'setcover':
        args.instances += [
            {'type': 'small', 'path': f"{args.samples_root_path}/setcover_temporal_32/instances/transfer_500r_1000c_0.05d/instance_{i + 1}.lp"}
            for i in range(20)]
        args.instances += [
            {'type': 'medium', 'path': f"{args.samples_root_path}/setcover_temporal_32/instances/transfer_1000r_1000c_0.05d/instance_{i + 1}.lp"}
            for i in range(20)]
        args.instances += [
            {'type': 'big', 'path': f"{args.samples_root_path}/setcover_temporal_32/instances/transfer_2000r_1000c_0.05d/instance_{i + 1}.lp"}
            for i in range(20)]

    elif args.problem == 'cauctions':
        args.instances += [
           {'type': 'small', 'path': f"{args.samples_root_path}/cauctions_temporal_32/instances/transfer_100_500/instance_{i + 1}.lp"} for i in
           range(20)]
        args.instances += [
            {'type': 'medium', 'path': f"{args.samples_root_path}/cauctions_temporal_32/instances/transfer_200_1000/instance_{i + 1}.lp"} for i
            in range(20)]
        args.instances += [
            {'type': 'big', 'path': f"{args.samples_root_path}/cauctions_temporal_32/instances/transfer_300_1500/instance_{i + 1}.lp"} for i in
            range(20)]

    elif args.problem == 'facilities':
        args.instances += [
            {'type': 'small', 'path': f"{args.samples_root_path}/facilities_temporal_32/instances/transfer_100_100_5/instance_{i + 1}.lp"} for i
            in range(20)]
        args.instances += [{'type': 'medium', 'path': f"{args.samples_root_path}/facilities_temporal_32/instances/transfer_200_100_5/instance_{i + 1}.lp"}
                           for i in range(20)]
        args.instances += [{'type': 'big', 'path': f"{args.samples_root_path}/facilities_temporal_32/instances/transfer_400_100_5/instance_{i + 1}.lp"} for
                           i in range(20)]

    elif args.problem == 'indset':
         args.instances += [{'type': 'small', 'path': f"{args.samples_root_path}/indset_temporal_32/instances/transfer_500_4/instance_{i + 1}.lp"} for i in
                            range(20)]
         args.instances += [{'type': 'medium', 'path': f"{args.samples_root_path}/indset_temporal_32/instances/transfer_1000_4/instance_{i + 1}.lp"} for i
                           in range(20)]
         args.instances += [{'type': 'big', 'path': f"{args.samples_root_path}/indset_temporal_32/instances/transfer_1500_4/instance_{i + 1}.lp"} for i in
                            range(20)]

    else:
        raise NotImplementedError

    ckpt_file = f'best_trained_params_{args.problem}_v{args.version}.pkl'
    args.cache_ckpt_path = os.path.join(args.dir_path, args.logs_dir, 'ckpt', ckpt_file)

    cache_results_file = f"{args.logs_dir}/results/{args.problem}_{args.timelimit}_{time.strftime('%Y-%m-%d__%H-%M-%S')}.csv"
    obs_results_file = f"{args.dir_path}/{args.logs_dir}/results/{args.problem}_{args.timelimit}_{time.strftime('%Y-%m-%d__%H-%M-%S')}.csv"

    seeds = [0]
    internal_branchers = ['relpscost', 'fullstrong', 'allfullstrong', 'vanillafullstrong']
    gnn_models = ['supervised'] # Can be supervised
    time_limit = args.timelimit

    branching_policies = []



    # SCIP internal brancher baselines
    for brancher in internal_branchers:
        for seed in seeds:
            branching_policies.append({
                    'type': 'internal',
                    'name': brancher,
                    'seed': seed,
             })

    # GNN models
    for model in gnn_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'gnn',
                'name': model,
                'seed': seed,
            })

    print(f"problem: {args.problem}")
    print(f"gpu: {args.device}")
    print(f"time limit: {time_limit} s")

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    loaded_calls = {}
    for policy in branching_policies:
        if policy['type'] == 'gnn':
            policy['model'] = TemporalGCNNEvaluatePolicy(args)

    print("running SCIP...")

    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'walltime',
        'proctime',
    ]

    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': time_limit,
                       'timing/clocktype': 1, 'branching/vanillafullstrong/idempotent': True}

    with open(cache_results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in args.instances:
            print(f"\n{instance['type']}: {instance['path']}...")
            for policy in branching_policies:
                try:
                    if policy['type'] == 'internal':
                        walltime_int_ = None

                        # Run SCIP's default brancher
                        env = ecole.environment.Configuring(scip_params={**scip_parameters,
                                                            f"branching/{policy['name']}/priority": 9999999})
                        env.seed(policy['seed'])

                        walltime = time.perf_counter()
                        proctime = time.process_time()

                        env.reset(instance['path'])
                        _, _, _, _, _ = env.step({})

                        walltime = time.perf_counter() - walltime
                        proctime = time.process_time() - proctime

                    elif policy['type'] == 'gnn':
                        # Run the GNN policy
                        observation_function = {"scores": ExploreThenStrongBranch(expert_probability=0.0),
                                                "node_observation": ecole.observation.NodeBipartite()}

                        env = ecole.environment.Branching(observation_function=observation_function,
                                                          scip_params=scip_parameters,
                                                          #pseudo_candidates=True
                                                          )
                        env.seed(policy['seed'])
                        torch.manual_seed(policy['seed'])

                        walltime = time.perf_counter()
                        proctime = time.process_time()

                        observation, sample_action_set, _, done, _ = env.reset(instance['path'])
                        iter = 0
                        gruInput = []
                        with torch.no_grad():
                            while iter <  args.GRU_temp_dim - 1:
                                scores, scores_are_expert = observation["scores"]
                                gruInput = policy['model'](sample_action_set, observation['node_observation'], gruInput)
                                action = sample_action_set[scores[sample_action_set].argmax()]
                                observation, sample_action_set, _, done, _ = env.step(action)
                                iter += 1

                            proctime_int_ = 0
                            walltime_int_ = 0
                            cnt = 0

                            while not done:
                                walltime_ = time.perf_counter()
                                proctime_ = time.process_time()

                                action, gruInput = policy['model'].branch(sample_action_set, observation['node_observation'], gruInput)

                                walltime_int_ += (time.perf_counter() - walltime_)
                                proctime_int_ += (time.process_time() - proctime_)

                                observation, sample_action_set, _, done, _ = env.step(action)

                                cnt+=1

                        walltime = time.perf_counter() - walltime
                        proctime = time.process_time() - proctime

                except Exception as e:
                    print(e,'red')

                scip_model = env.model.as_pyscipopt()
                stime = scip_model.getSolvingTime()
                nnodes = scip_model.getNNodes()
                nlps = scip_model.getNLPs()
                gap = scip_model.getGap()
                status = scip_model.getStatus()

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'walltime': walltime,
                    'proctime': proctime,
                })
                csvfile.flush()
                print(
                    f"  {policy['type']}:{policy['name']} gap: {gap} - nodes: {nnodes}  lps: {nlps}  stime: {stime:.2f} (wall: {walltime:.2f}  proc: {proctime:.2f}) status: {status}")
                if walltime_int_:
                    print(f'avg_wall_time_T {walltime_int_ / cnt}    avg_proc_time_T: {proctime_int_ / cnt}', 'green')

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

    get_metrics(cache_results_file,time_limit=time_limit)



def get_metrics(results_csv_path, time_limit=1*60):
    df = pd.read_csv(results_csv_path)
    #print(df.keys())
    x = df['policy']
    t = df['type']
    n = df['nnodes']
    ti = df['stime']
    g = df['gap']

    s_index_int = [i for i, t in t.items() if t=='small' and 'internal' in x[i] ]
    s_index_G = [i for i, t in t.items() if t=='small' and 'supervised' in x[i]]

    m_index_int = [i for i, t in t.items() if t == 'medium'  and 'internal' in x[i]]
    m_index_G = [i for i, t in t.items() if t == 'medium' and 'supervised' in x[i]]

    b_index_int = [i for i, t in t.items() if t == 'big'  and 'internal' in x[i]]
    b_index_G = [i for i, t in t.items() if t == 'big' and 'supervised' in x[i]]


    s_nnodes_GCN  = n[s_index_G].mean()
    s_nnodes_int = n[s_index_int].mean()

    m_nnodes_GCN = n[m_index_G].mean()
    m_nnodes_int = n[m_index_int].mean()

    b_nnodes_GCN = n[b_index_G].mean()
    b_nnodes_int = n[b_index_int].mean()



    s_gap_GCN  = g[s_index_G].mean()
    s_gap_int = g[s_index_int].mean()

    m_gap_GCN = g[m_index_G].mean()
    m_gap_int = g[m_index_int].mean()

    b_gap_GCN = g[b_index_G].mean()
    b_gap_int = g[b_index_int].mean()



    s_ti_GCN  = ti[s_index_G].mean()
    s_ti_int = ti[s_index_int].mean()

    m_ti_GCN = ti[m_index_G].mean()
    m_ti_int = ti[m_index_int].mean()

    b_ti_GCN = ti[b_index_G].mean()
    b_ti_int = ti[b_index_int].mean()

    s_win_GCN = ti[s_index_G] - ti[s_index_int]
    print('------------------------NUM NODES-------------------------','red')
    print(f'nnodes_int: {s_nnodes_int} {m_nnodes_int} {b_nnodes_int}')
    print(f'nnodes_GCN: {s_nnodes_GCN} {m_nnodes_GCN} {b_nnodes_GCN}')

    print('------------------------AVG GAP-------------------------', 'red')
    print(f'gap_int: {s_gap_int:.3f} {m_gap_int:.3f} {b_gap_int:.3f}')
    print(f'gap_GCN: {s_gap_GCN:.3f} {m_gap_GCN:.3f} {b_gap_GCN:.3f}')

    print('------------------------AVG TIME--------------------------','red')
    print(f'avg_time_int: {s_ti_int:.2f} {m_ti_int:.2f} {b_ti_int:.2f}')
    print(f'avg_time_GCN: { s_ti_GCN:.2f} {m_ti_GCN:.2f} {b_ti_GCN:.2f}')
    print('------------------------NUM WINS--------------------------','red')
    print(f'win small: {sum((np.sign(ti[s_index_int].to_numpy() -ti[s_index_G].to_numpy()) + 1)/2)}/{len(ti[s_index_G].to_numpy())}      Percent: {sum((np.sign(ti[s_index_int].to_numpy()-ti[s_index_G].to_numpy()) + 1)/2)/len(ti[s_index_G].to_numpy()) *100 :.2f}%')
    print(f'win medium: {sum((np.sign(ti[m_index_int].to_numpy() - ti[m_index_G].to_numpy()) + 1) / 2)}/{len(ti[m_index_G].to_numpy())} Percent: {sum((np.sign(ti[m_index_int].to_numpy() - ti[m_index_G].to_numpy()) + 1) / 2)/len(ti[m_index_G].to_numpy()) *100:.2f}%')
    print(f'win big: {sum((np.sign(ti[b_index_int].to_numpy() - ti[b_index_G].to_numpy()) + 1) / 2)}/{len(ti[b_index_G].to_numpy())}    Percent: {sum((np.sign(ti[b_index_int].to_numpy() - ti[b_index_G].to_numpy()) + 1) / 2)/len(ti[b_index_G].to_numpy()) * 100 :.2f}%')
    print('-----------------------------------------------------------')


if __name__ == "__main__":
    main()
