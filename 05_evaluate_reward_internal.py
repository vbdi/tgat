import argparse
import csv
import json
import pathlib
import ecole as ec
import numpy as np
import torch
import moxing as mox
from constants import *
import pandas as pd
from environments import Branching as Environment
from rewards import TimeLimitDualIntegral as BoundIntegral
from rewards import NodeDualIntegral as NodeIntegral
from utilities import *
import os
from datetime import datetime
epsilon = 1e-6


def main(args):
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

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu !=-1 else 'cpu')
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
                          for i in range( 20)]
        args.instances += [{'type': 'big', 'path': f"{args.samples_root_path}/facilities_temporal_32/instances/transfer_400_100_5/instance_{i + 1}.lp"} for
                           i in range(20)]

    elif args.problem == 'indset':
        args.instances += [{'type': 'small', 'path': f"{args.samples_root_path}/indset_temporal_32/instances/transfer_500_4/instance_{i + 1}.lp"} for i in
                           range(20)]
        args.instances += [{'type': 'medium', 'path': f"{args.samples_root_path}/indset_temporal_32/instances/transfer_1000_4/instance_{i + 1}.lp"} for i
                           in range(20)]
        args.instances += [{'type': 'big', 'path': f"{args.samples_root_path}/indset_temporal_32/instances/transfer_1500_4/instance_{i + 1}.lp"} for i in
                           range(20)]

    elif args.problem == 'item_placement':
        args.instances += [{'type': 'mixed',
                        'path': f"{args.samples_root_path}/item_placement_temporal_32/instances/1_item_placement"
                                f"/test/item_placement_{i + 1}.mps.gz"} for i in range(9999, 10099)]

    elif args.problem == 'load_balancing':
        args.instances += [{'type': 'mixed',
                        'path': f"{args.samples_root_path}/load_balancing_temporal_4/instances/2_load_balancing"
                                f"/test/load_balancing_{i + 1}.mps.gz"} for i in range(9999, 10099)]

    elif args.problem == 'anonymous':
        args.instances += [{'type': 'mixed',
                        'path': f"{args.samples_root_path}/anonymous_temporal_32/instances/3_anonymous"
                                f"/test/anonymous_{i + 1}.mps.gz"} for i in range(118, 138)]
    else:
        NotImplementedError

    ckpt_file = f'best_trained_params_{args.problem}_v{args.version}.pkl'
    args.cache_ckpt_path = os.path.join(args.dir_path, args.logs_dir, 'ckpt', ckpt_file)

    results_fieldnames = ['agent', 'instance', 'seed', 'objective_offset','cumulated_reward', 'stime', 'nnodes', 'nlps', 'gap', 'status']
    outcome = {'agent': [], 'time_limit':[], 'reward_time':[],  'nnodes':[],'iter':[], 'stime':[], 'gap':[]}

    import sys
    sys.path.insert(1, str(pathlib.Path.cwd()))
    timelimit = [3600, 2400, 1200, 900, 480, 240, 120, 60]
    for time_limit in timelimit:
        for agent in ["FSB","PSE","RTB","Random"]:
            args.agent = agent

            processing_time = []
            # override from command-line argument if provided
            integral_function = BoundIntegral()
            node_integral_function = NodeIntegral()

            dd = datetime.today().strftime('%Y-%m-%d')
            dt = datetime.today().strftime('%H-%M-%S')

            cache_results_file = f"{args.logs_dir}/results/{args.problem}_{time_limit}_{args.agent}_" \
                                 f"_{dd}_{dt}.csv"
            obs_results_file = f"{args.dir_path}/{args.logs_dir}/results/{args.problem}_{time_limit}_{args.agent}_" \
                               f"_{dd}_{dt}.csv"

            with open(cache_results_file, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
                writer.writeheader()

            # set up the proper agent, environment and goal for the task
            if args.agent == "Random":
                policy = Random()
                observation_function = ObservationFunction()
            elif args.agent == "PPO":
                raise NotImplementedError("Not implemented")
            elif args.agent == "GCNN":
                policy = GCNNEvaluatePolicy(args)
                observation_function = ObservationFunction()

            elif args.agent == 'NORM':
                policy = GCNNEvaluatePolicyNorm(args)
                observation_function = ObservationFunction()
            elif args.agent == "FSB":
                policy = General()
                observation_function = FSBObservation()

            elif args.agent == "PSE":
                policy = General()
                observation_function = Psebservation()


            elif args.agent == "RTB":
                scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': time_limit,
                                   'timing/clocktype': 1, 'branching/vanillafullstrong/idempotent': True}

                env = ec.environment.Configuring(scip_params={**scip_parameters,
                                                           f'branching/relpscost/priority': 9999999},
                                                 reward_function=[-integral_function, -node_integral_function])

                # evaluation loop
                for seed, instance in enumerate(args.instances):
                    try:
                        # seed both the agent and the environment (deterministic behavior)
                        env.seed(seed)
                        cumulated_reward = 0  # discard initial reward
                        objective_offset = 0
                        node_cumulated_reward = 0
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

                        integral_function.set_parameters(
                            initial_primal_bound=initial_primal_bound,
                            initial_dual_bound=initial_dual_bound,
                            objective_offset=objective_offset)
                        node_integral_function.set_parameters(
                            initial_primal_bound=initial_primal_bound,
                            initial_dual_bound=initial_dual_bound,
                            objective_offset=objective_offset)

                        print('')
                        print(f"  instance {instance['path']}")
                        print(f"  scip_params: {scip_parameters}")
                        print(f"  seed: {seed} / {len(args.instances)}")
                        print(f"  initial primal bound: {initial_primal_bound}")
                        print(f"  initial dual bound: {initial_dual_bound}")
                        print(f"  objective offset: {objective_offset}")
                        print(f"  time_limit: {time_limit}")
                        print(f"  agent: {args.agent}")
                        # reset the environment
                        env.reset(instance['path'])
                        _, _, reward, _, _ = env.step({})

                        cumulated_reward += reward[0]
                        node_cumulated_reward += reward[1]

                        scip_model = env.model.as_pyscipopt()
                        stime = scip_model.getSolvingTime()
                        nnodes = scip_model.getNNodes()
                        nlps = scip_model.getNLPs()
                        gap = scip_model.getGap()
                        status = scip_model.getStatus()
                        total_time = scip_model.getParam("limits/time")
                        processing_time.append(stime)
                        print(
                            f"  cumulated reward (to be maximized): {cumulated_reward:.2f}\t "
                            f" stime: "
                            f"{stime:.2f} nodes_solved: {nnodes} gap: {gap:.2e} status: {status} ")

                        # save instance results
                        with open(cache_results_file, mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
                            writer.writerow({
                                'agent': args.agent,
                                'instance': instance['path'],
                                'seed': seed,
                                # 'initial_primal_bound': initial_primal_bound,
                                # 'initial_dual_bound': initial_dual_bound,
                                'objective_offset': objective_offset,
                                'cumulated_reward': cumulated_reward,
                                'stime': stime,
                                'nnodes': nnodes,
                                'nlps': nlps,
                                'gap': gap,
                                'status': status,
                            })

                    except Exception as e:
                        print(f'error: {e}', 'red')
                        # save instance results
                        with open(cache_results_file, mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
                            writer.writerow({
                                'agent': args.agent,
                                'instance': instance['path'],
                                'seed': seed,
                                # 'initial_primal_bound': initial_primal_bound,
                                # 'initial_dual_bound': initial_dual_bound,
                                'objective_offset': objective_offset,
                                'cumulated_reward': 0,
                                'stime': 0,
                                'nnodes': 0,
                                'nlps': 0,
                                'gap': 0,
                                'status': 'error',
                            })


            if args.agent != "RTB":
                env = Environment(
                    time_limit=time_limit,
                    observation_function=observation_function,
                    reward_function=[-integral_function, -node_integral_function],  # negated integral (minimization)
                )


                # evaluation loop
                for seed, instance in enumerate(args.instances):
                    try:
                        # seed both the agent and the environment (deterministic behavior)
                        observation_function.seed(seed)
                        policy.seed(seed)
                        env.seed(seed)
                        cumulated_reward = 0  # discard initial reward
                        objective_offset = 0
                        node_cumulated_reward = 0
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

                        integral_function.set_parameters(
                                initial_primal_bound=initial_primal_bound,
                                initial_dual_bound=initial_dual_bound,
                                objective_offset=objective_offset)
                        node_integral_function.set_parameters(
                            initial_primal_bound=initial_primal_bound,
                            initial_dual_bound=initial_dual_bound,
                            objective_offset=objective_offset)

                        print('')
                        print(f"  instance {instance['path']}")
                        print(f"  seed: {seed} / {len(args.instances)}")
                        print(f"  initial primal bound: {initial_primal_bound}")
                        print(f"  initial dual bound: {initial_dual_bound}")
                        print(f"  objective offset: {objective_offset}")
                        print(f"  time_limit: {time_limit}")
                        print(f"  agent: {args.agent}")


                        # reset the environment
                        observation, action_set, reward, done, info = env.reset(instance['path'], objective_limit=initial_primal_bound)

                        if args.debug:
                            print(f"  info: {info}")
                            print(f"  reward: {reward}",'red')
                            print(f"  action_set: {action_set}")
                        # loop over the environment
                        iter = 0
                        while not done:
                            action = policy(action_set, observation)
                            observation, action_set, reward, done, info = env.step(action)

                            if args.debug:
                                print(f"  action: {action}")
                                print(f"  info: {info}")
                                print(f"  reward: {reward}")
                                print(f"  action_set: {action_set}")

                            cumulated_reward += reward[0]
                            node_cumulated_reward += reward[1]
                            iter+=1

                        scip_model = env.model.as_pyscipopt()
                        stime = scip_model.getSolvingTime()
                        nnodes = scip_model.getNNodes()
                        nlps = scip_model.getNLPs()
                        gap = scip_model.getGap()
                        status = scip_model.getStatus()
                        total_time = scip_model.getParam("limits/time")
                        processing_time.append(stime)



                        print(f"  cumulated reward (to be maximized): {cumulated_reward:.2f}\t  stime: "
                              f"{stime:.2f} nodes_solved: {nnodes} gap: {gap:.2e} status: {status} ")

                        # save instance results
                        with open(cache_results_file, mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
                            writer.writerow({
                                'agent': args.agent,
                                'instance': instance['path'],
                                'seed': seed,
                                # 'initial_primal_bound': initial_primal_bound,
                                # 'initial_dual_bound': initial_dual_bound,
                                'objective_offset': objective_offset,
                                'cumulated_reward': cumulated_reward,
                                'stime': stime,
                                'nnodes': nnodes,
                                'nlps': nlps,
                                'gap': gap,
                                'status': status,
                            })

                    except Exception as e:
                        print(f'error: {e}', 'red')
                        # save instance results
                        with open(cache_results_file, mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
                            writer.writerow({
                                'agent': args.agent,
                                'instance': instance['path'],
                                'seed': seed,
                                # 'initial_primal_bound': initial_primal_bound,
                                # 'initial_dual_bound': initial_dual_bound,
                                'objective_offset': objective_offset,
                                'cumulated_reward': cumulated_reward,
                                'stime': stime,
                                'nnodes': nnodes,
                                'nlps': 0,
                                'gap': 0,
                                'status': 'error',
                            })

            avg_initial_primal_bound, avg_initial_dual_bound, avg_objective_offset, avg_cumulated_reward, cum_nnodes, avg_stime, avg_gap = get_metrics(
                args, cache_results_file, time_limit=time_limit)
            outcome['agent'].append(args.agent)
            outcome['time_limit'].append(time_limit)
            outcome['reward_time'].append(avg_cumulated_reward)
            outcome['nnodes'].append(cum_nnodes)
            outcome['iter'].append(iter - 1)
            outcome['stime'].append(avg_stime)
            outcome['gap'].append(avg_gap)

            print(
                '-----------------------------------------SUMMARY------------------------------------------------')
            LEN = len(outcome['reward_time'])
            KEYS = outcome.keys()
            for k0 in range(LEN):
                for k1 in KEYS:
                    if isinstance(outcome[k1][k0], str):
                        print('{:<2}:  {}'.format(k1, outcome[k1][k0][0:3]), end='  ')
                    elif k1 == 'gap':
                        print('{:<2}:  {:.2e}'.format(k1, outcome[k1][k0]), end='  ')
                    elif k1 in ['stime', 'expert_prob', 'time_limit']:
                        print('{:<2}:  {:.1f}'.format(k1, outcome[k1][k0]), end='    ')
                    else:
                        print('{:<2}:  {:.1f}'.format(k1, outcome[k1][k0]), end='    ')
                print('\n')

            print(f'--------------------------------------------{args.agent}:{time_limit}--------------------------------------------------')
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
    agent = df['agent'].values
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
        s_time_GCN = (tt[s_index_G] + 1).prod()**(1.0/len(s_index_G))
    else:
        s_time_GCN = 0

    if len(m_index_G) > 0:
        m_time_GCN = (tt[m_index_G] + 1).prod()**(1.0/len(m_index_G))
    else:
        m_time_GCN = 0
    if len(b_index_G) > 0:
        b_time_GCN = (tt[b_index_G] + 1).prod()**(1.0/len(b_index_G))
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
                        choices=['setcover','cauctions', 'item_placement', 'load_balancing', 'anonymous', 'indset', 'facilities'])
    parser.add_argument('--samples_path', type=str, default='data/samples/')
    parser.add_argument('--logs_dir', type=str, default='logs/')
    parser.add_argument('-s', '--seed', help='Random generator seed.', type=int, default=0)
    parser.add_argument('--node_coef', help='node reward coef', type=int, default=400000)
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--version', type=str)
    parser.add_argument('--use_roma', help='use roma', action='store_true')


    # ROMA related args
    parser.add_argument('--dir_path', default='.')

    # Bipartitde graph parameters
    parser.add_argument('--emb_size', default=64, type=int)

    # evaluation parameters
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
        default="GCNN",
        type=str,
        choices=("Random", "Temporal", "GCNN", "FSB","PSE", "RTB"),
    )

    args, unparsed = parser.parse_known_args()
    main(args)

#python evaluate_challenge.py --problem setcover --agent GCNN