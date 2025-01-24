import os
import pandas as pd
import argparse
import numpy as np

if os.getcwd().split('/')[2] == 'mehdi':
    from PColor import ColorPrint
    print = ColorPrint().print

def get_metrics(results_csv_path, time_limit=1*60):
    df = pd.read_csv(results_csv_path)

    t = df['instance']
    r = df['cumulated_reward']


    s_index_G = [i for i, t in t.items() if '100_500' in t ]
    m_index_G = [i for i, t in t.items() if '200_1000' in t]
    b_index_G = [i for i, t in t.items() if '300_1500' in t]

    s_nnodes_GCN  = r[s_index_G].mean()
    m_nnodes_GCN = r[m_index_G].mean()
    b_nnodes_GCN = r[b_index_G].mean()


    print('------------------------REWARD-------------------------\n','red')
    print(f'nnodes_GCN: {s_nnodes_GCN}\t{m_nnodes_GCN}\t{b_nnodes_GCN}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--problem', type=str, default='setcover', help='MILP instance type',
    #                     choices=['setcover', 'cauctions', 'item_placement', 'load_balancing', 'anonymous', 'indset',
    #                              'facilities'])
    # parser.add_argument('--dir_path', type=str, default='logs')
    # parser.add_argument('--version', type=int)
    # parser.add_argument('--csv', type=str)
    parser.add_argument('path', type=str)
    args, unparsed = parser.parse_known_args()

    print(args.path,'blue')
    #results_csv_path = os.path.join(args.dir_path,f'v{args.version}',args.problem,'results',args.csv)
    get_metrics(args.path)

