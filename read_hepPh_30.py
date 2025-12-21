"""
http://networkrepository.com/ca-cit-HepPh.php
Size	28,093 vertices (authors)
Volume	4,596,803 edges (collaborations)
Unique volume	3,148,447 edges (collaborations)
Average degree	327.26 edges / vertex
2337 graph // 30== 80 graph snapshots, we fetch
the latest [-36,-1] i.e. 36 graph (2 years) except the last one
undirected without self-loop dynamic graphs and isolated nodes
total graphs:  41
we take out and save the last 36 graphs......
36
@ graph 0 # of nodes 1928 # of edges 35756
@ graph 1 # of nodes 2026 # of edges 29418
@ graph 2 # of nodes 1775 # of edges 22589
@ graph 3 # of nodes 2060 # of edges 28391
@ graph 4 # of nodes 1952 # of edges 32081
@ graph 5 # of nodes 1779 # of edges 24282
@ graph 6 # of nodes 1906 # of edges 40858
@ graph 7 # of nodes 2219 # of edges 30576
@ graph 8 # of nodes 2279 # of edges 27949
@ graph 9 # of nodes 2684 # of edges 46250
@ graph 10 # of nodes 2244 # of edges 45806
@ graph 11 # of nodes 2205 # of edges 25457
@ graph 12 # of nodes 2595 # of edges 42692
@ graph 13 # of nodes 2842 # of edges 73819
@ graph 14 # of nodes 2121 # of edges 41253
@ graph 15 # of nodes 2410 # of edges 50776
@ graph 16 # of nodes 2551 # of edges 39998
@ graph 17 # of nodes 2080 # of edges 27764
@ graph 18 # of nodes 2554 # of edges 31564
@ graph 19 # of nodes 2473 # of edges 51842
@ graph 20 # of nodes 2757 # of edges 43353
@ graph 21 # of nodes 2473 # of edges 36405
@ graph 22 # of nodes 2901 # of edges 43125
@ graph 23 # of nodes 2806 # of edges 46910
@ graph 24 # of nodes 2839 # of edges 52983
@ graph 25 # of nodes 3028 # of edges 69435
@ graph 26 # of nodes 2486 # of edges 44736
@ graph 27 # of nodes 2780 # of edges 43833
@ graph 28 # of nodes 3010 # of edges 50851
@ graph 29 # of nodes 2760 # of edges 45525
@ graph 30 # of nodes 3784 # of edges 92636
@ graph 31 # of nodes 3373 # of edges 71714
@ graph 32 # of nodes 3480 # of edges 62923
@ graph 33 # of nodes 3080 # of edges 60947
@ graph 34 # of nodes 3324 # of edges 56914
@ graph 35 # of nodes 3320 # of edges 60170
time gap is 30
total edges: 976097
total nodes: 15330


"""

import datetime
import pickle
import torch
import networkx as nx
import pandas as pd

gap = 30

def save_nx_graph(nx_graph, path='dyn_graphs_dataset.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file

def save_edges(edges, dataset):
    path = '../input/processed/{}30/{}30'.format(dataset, dataset)
    torch.save(edges, path+'.pt')
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(edges, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file
    print('saved!')

def getID(node_id, nodes_dict):
    if node_id not in nodes_dict.keys():
        idx = len(nodes_dict)
        nodes_dict[node_id] = idx
    else:
        idx = nodes_dict[node_id]
    return idx, nodes_dict

if __name__ == '__main__':
    # --- load dataset ---
    dataset = 'HepPh'
    path = '../input/raw/{}'.format(dataset)
    df = pd.read_csv(path, sep=' | \t', names=['from', 'to', 'weight', 'time'], header=None, comment='%',
                     engine='python')
    df['time'] = df['time'].apply(lambda x: int(datetime.datetime.utcfromtimestamp(x).strftime('%Y%m%d')))
    all_days = len(pd.unique(df['time']))
    print('# of all edges: ', len(df))
    print('all unique days: ', all_days)
    print(df.head(5))

    # --- check the time oder, if not ascending, resort it ---
    tmp = df['time'][0]
    for i in range(len(df['time'])):
        if df['time'][i] > tmp:
            tmp = df['time'][i]
        elif df['time'][i] == tmp:
            pass
        else:
            print('not ascending --> we resorted it')
            print(df[i - 2:i + 2])
            df.sort_values(by='time', ascending=True, inplace=True)
            df.reset_index(inplace=True)
            print(df[i - 2:i + 2])
            break
        if i == len(df['time']) - 1:
            print('ALL checked --> ascending!!!')

    # --- generate graph and dyn_graphs ---
    cnt_graphs = 0
    graphs = []
    g = nx.Graph()
    tmp = df['time'][0]  # time is in ascending order
    for i in range(len(df['time'])):
        if tmp == df['time'][i]:  # if is in current day
            g.add_edge(str(df['from'][i]), str(df['to'][i]))
            if i == len(df['time']) - 1:  # EOF ---
                cnt_graphs += 1
                # graphs.append(g.copy())  # ignore the last day
                print('processed graphs ', cnt_graphs, '/', all_days, 'ALL done......\n')
        elif tmp < df['time'][i]:  # if goes to next day
            cnt_graphs += 1
            if (cnt_graphs // gap) >= (
                    all_days // gap - 40) and cnt_graphs % gap == 0:  # the last 50 graphs 'and' the gap
                g.remove_edges_from(g.selfloop_edges())
                g.remove_nodes_from(list(nx.isolates(g)))
                graphs.append(g.copy())  # append previous g; for a part of graphs to reduce ROM
                g = nx.Graph()            # reset graph, based on the real-world application
            if cnt_graphs % 50 == 0:
                print('processed graphs ', cnt_graphs, '/', all_days)
            tmp = df['time'][i]
            g.add_edge(str(df['from'][i]), str(df['to'][i]))
        else:
            print('ERROR -- EXIT -- please double check if time is in ascending order!')
            exit(0)

    # --- take out and save part of graphs ----
    print('total graphs: ', len(graphs))
    print('we take out and save the last 36 graphs......')
    # raw_graphs = graphs[-22:-1]  # the last graph has some problem... we ignore it!
    raw_graphs = graphs[-37:-1]  # the last graph has some problem... we ignore it!
    print(len(raw_graphs))
    # remap node index:
    G = nx.Graph() # whole graph, to count number of nodes and edges
    graphs = [] # graph list, to save remapped graphs
    nodes_dict = {} # node re-id index, to save mapped index
    edges_list = [] # edge_index lsit, sparse matrix
    for i, raw_graph in enumerate(raw_graphs):
        g = nx.Graph()
        for edge in raw_graph.edges:
            idx_i, nodes_dict = getID(edge[0], nodes_dict)
            idx_j, nodes_dict = getID(edge[1], nodes_dict)
            g.add_edge(idx_i, idx_j)
        graphs.append(g) # append to graph list
        edges_list.append(list(g.edges)) # append to edge list
        G.add_edges_from(g.edges) # append to the whole graphs
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('time gap is {}'.format(gap))
    print('total edges: {}'.format(G.number_of_edges()))
    print('total nodes: {}'.format(G.number_of_nodes()))
    save_edges(edges_list, dataset)
    print(max(nodes_dict.values())+1)