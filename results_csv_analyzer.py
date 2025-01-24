import os
import pandas as pd
import argparse
import numpy as np

if os.getcwd().split('/')[2] == 'mehdi':
    from PColor import ColorPrint
    print = ColorPrint().print

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
    print(f'gap_int: {s_gap_int:.4f} {m_gap_int:.4f} {b_gap_int:.4f}')
    print(f'gap_GCN: {s_gap_GCN:.4f} {m_gap_GCN:.4f} {b_gap_GCN:.4f}')

    print('------------------------AVG TIME--------------------------','red')
    print(f'avg_time_int: {s_ti_int:.2f} {m_ti_int:.2f} {b_ti_int:.2f}')
    print(f'avg_time_GCN: { s_ti_GCN:.2f} {m_ti_GCN:.2f} {b_ti_GCN:.2f}')
    print('------------------------NUM WINS--------------------------','red')

    print(f'{len(s_index_G)} {len(s_index_int)}','yellow')
    print(f'win small: {sum((np.sign(ti[s_index_int].to_numpy() -ti[s_index_G].to_numpy()) + 1)/2)}/{len(ti[s_index_G].to_numpy())}      Percent: {sum((np.sign(ti[s_index_int].to_numpy()-ti[s_index_G].to_numpy()) + 1)/2)/len(ti[s_index_G].to_numpy()) *100 :.2f}%')
    print(f'win medium: {sum((np.sign(ti[m_index_int].to_numpy() - ti[m_index_G].to_numpy()) + 1) / 2)}/{len(ti[m_index_G].to_numpy())} Percent: {sum((np.sign(ti[m_index_int].to_numpy() - ti[m_index_G].to_numpy()) + 1) / 2)/len(ti[m_index_G].to_numpy()) *100:.2f}%')
    print(f'win big: {sum((np.sign(ti[b_index_int].to_numpy() - ti[b_index_G].to_numpy()) + 1) / 2)}/{len(ti[b_index_G].to_numpy())}    Percent: {sum((np.sign(ti[b_index_int].to_numpy() - ti[b_index_G].to_numpy()) + 1) / 2)/len(ti[b_index_G].to_numpy()) * 100 :.2f}%')
    print('-----------------------------------------------------------')


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

