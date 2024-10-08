import numpy as np


def dijkstra_paths(adj, indeg = [0]):
    try:
        import networkx as nx
    except ImportError:
        print("please install networkx first")
    G = nx.convert_matrix.from_numpy_matrix(A = np.where(adj == np.inf, 0, adj), create_using=nx.DiGraph)
    starts = [x for x,d in G.in_degree() if d in indeg]
    # # also can use
    # dist, paths = nx.multi_source_dijkstra(G, starts, weight = 'weight')
    # the output format need to be rearranged

    # complexity o(n2)
    paths = [nx.shortest_path(G, x, weight = 'weight') for x,d in G.in_degree() if d in indeg]
    # paths = [nx.single_source_dijkstra_path(G, x, weight='weight') for x,d in G.in_degree() if d == 0]
    
    results = {}
    opt = np.array([[np.inf] * adj.shape[0]] * adj.shape[1])
    # complexity o(n2)
    for i, path in enumerate(paths):
        start = starts[i]
        for end, values in path.items():
            results[str(start)+"_"+str(end)] = values
            opt[start,end] = 0
            if len(values) > 1:
                for j, u in enumerate(values[:-1]):
                    opt[start,end] = opt[start,end] + G[u][values[j+1]]['weight'] 
    return results,opt


def Floyd_warshall(adj):
    try:
        import networkx as nx
    except ImportError:
        print("please install networkx first")
    else:
        # consider disconnected nodes as connected with np.inf weight, or disconnected graph causing error
        G = nx.convert_matrix.from_numpy_matrix(A = adj, create_using = nx.DiGraph)
        predecessors, distance = nx.floyd_warshall_predecessor_and_distance(G)
        N = adj.shape[0]
        paths = {}
        for i in range(N):
            for j in range(N):
                paths[str(i)+"_"+str(j)] = nx.reconstruct_path(i, j, predecessors)
        return paths, distance

#save memory but same results
def floyd_warshall(adj):
    try:
        import networkx as nx
        import numpy as np
    except ImportError:
        print("Please install networkx and numpy.")
        return None

    # Directly create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    # Use floyd_warshall_predecessor_and_distance directly
    predecessors, distance = nx.floyd_warshall_predecessor_and_distance(G)
    
    paths = {}
    N = adj.shape[0]
    for i in range(N):
        for j in range(N):
            if i != j:
                try:
                    # Reconstruct the path from i to j
                    path = nx.reconstruct_path(i, j, predecessors)
                    paths[f"{i}_{j}"] = path
                except KeyError:
                    # Handle the case where there is no path from i to j
                    paths[f"{i}_{j}"] = "No path"

    return paths, distance

    
def _greedy_path_step(paths, max_w, opt_value, nodes_visit, length_bias = None):
    """\
    greedily choose trajactory paths, one step.

    Parameters
    ----------
    paths
        The output of floydWarshall algorithm, dictionary including all-pair shortest path
    adj_matrix
        adjacency matrix of the graph
    opt_value
        The output of floydWarshall algorithm, the optimal value for all-pair shortest path
    node_visit
        List that store the coverd nodes
    length_bias
        The bias on the path length for greedy selection
    Returns
    -------
    max_idx
        The path index picked
    max_cover
        The newly covered nodes with max_idx chosen
    """
    max_idx = None
    max_cover = 0

    if length_bias == None:
        covers = [[i, np.sum(1 - nodes_visit[paths[i]])] for i in paths.keys()]
    # with length penalty
    else:
        if length_bias <= 1:  
            covers = [[i, np.sum(1 - nodes_visit[paths[i]]) + length_bias * len(paths[i])] for i in paths.keys()]
        else:
            covers = [[i, len(paths[i])] for i in paths.keys()]
    
    covers.sort(key=lambda x:x[1],reverse=True)

    for idx, val in covers:
        path = paths[idx]
        max_idx = idx
        if length_bias == None:
            max_cover = int(val)
        else:
            if length_bias <= 1: 
                max_cover = int(val - length_bias * len(path))
            else:
                max_cover = int(np.sum(1 - nodes_visit[paths[idx]]))
        break 

    if max_cover != 0:
        nodes_visit[np.array(paths[max_idx])] = 1
    return max_idx, max_cover   

def g_selection(nodes, paths, opt_value, adj_matrix = None, threshold = 0.6, max_w = None, cut_off = None, verbose = True, length_bias = None, max_trajs = None):
    """\
    greedily choose trajactory paths, and return the paths selected 

    Parameters
    ----------
    nodes
        number of nodes in the graph
    paths
        The output of floydWarshall algorithm, dictionary including all-pair shortest path
    opt_value
        The output of floydWarshall algorithm, the optimal value for all-pair shortest path
    threshold
        parameter for quality control
    max_w
        Maximal weight of adj_matrix, provide either adj_matrix or max_w
    cut_off
        Minimal number of cell in path
    verbose
        Output intermidate result
    length_bias
        The bias on the path length for greedy selection
    max_trajs
        The maximal number of trajectories output

    Returns
    -------
    greedy_paths
        List store the path index picked in order
    paths
        dictionary including all-pair shortest path
    """

    # eliminate cut_off
    filtered_paths = {}
    
    """
    if cut_off != None:
        print("cut off small paths and conduct quality control")
        filtered_paths = {x:y for x,y in paths.items() if (len(y) > cut_off) and (opt_value[y[0]][y[-1]]/len(y) < threshold * max_w)}
    else:
        print("conduct quality control")
        filtered_paths = {x:y for x,y in paths.items() if (opt_value[y[0]][y[-1]]/len(y) < threshold * max_w)}
    """

    print("conduct quality control")
    for x,y in paths.items():
        if len(y) > 0: 
            ave_weight = opt_value[y[0]][y[-1]]/len(y) 
        else:
            ave_weight = 0
        
        if ave_weight < threshold * max_w:
            if cut_off == None:
                filtered_paths[x] = y
            elif len(y) > cut_off:
                filtered_paths[x] = y
            
    
    paths_tmp = filtered_paths.copy()
    path_cover = nodes

    nodes_visit = np.zeros(nodes)
    greedy_paths = []
    
    print("selected path (starting_ending):")

    traj_count = 0

    while path_cover > 0:

        path_idx, path_cover = _greedy_path_step(paths = paths_tmp, max_w = max_w, opt_value = opt_value, nodes_visit = nodes_visit, length_bias = length_bias)

        if path_cover != 0:
            del paths_tmp[path_idx]
            if verbose:
                print('start_end: ', path_idx,', len: ', len(filtered_paths[path_idx]), 'newly covered:', path_cover, end = '\n')
        # maximal output        
        if max_trajs != None:
            traj_count += 1
            if traj_count >= max_trajs:
                break
                
        greedy_paths.append(path_idx)
    greedy_paths = greedy_paths[:-1]
    print("Finished")
    return greedy_paths, filtered_paths


def dp_selection(nodes, paths, opt_value, threshold = 0.62, cut_off = None, verbose = True, max_trajs = None):

    dp = [[float('inf')] * nodes for _ in range(1 << nodes)]
    parent = [[-1] * nodes for _ in range(1 << nodes)]
    

    for i in range(nodes):
        dp[1 << i][i] = 0 

    for mask in range(1 << nodes):
        for u in range(nodes):
            if mask & (1 << u):
                for v in range(nodes):
                    if mask & (1 << v) == 0 and opt_value[u][v] < threshold:
                        next_mask = mask | (1 << v)
                        if dp[next_mask][v] > dp[mask][u] + opt_value[u][v]:
                            dp[next_mask][v] = dp[mask][u] + opt_value[u][v]
                            parent[next_mask][v] = u
    
    all_nodes_mask = (1 << nodes) - 1
    min_cost, end_node = min((cost, node) for node, cost in enumerate(dp[all_nodes_mask]))
    
    dp_paths = []
    mask = all_nodes_mask
    while end_node != -1:
        dp_paths.append(end_node)
        next_node = parent[mask][end_node]
        mask &= ~(1 << end_node)  
        end_node = next_node
    dp_paths.reverse()  
    
    if verbose:
        print(f"动态规划选出的路径为：{dp_paths}，总成本为：{min_cost}")
    
    return dp_paths

# 示例参数
# nodes = 4  # 假设有4个节点
# paths = {}  # 假设paths是通过Floyd-Warshall算法计算出的所有节点对之间的最短路径
# opt_value = {}  # 假设opt_value是通过Floyd-Warshall算法计算出的最短路径的权重

# 调用动态规划路径选择函数
# dp_paths = dynamic_path_selection(nodes, paths, opt_value)
    
def print_score(slected_paths, paths, adj_matrix, paths_num = None, normalizor = None):
    """\
    Print the score/average_weight of all selected paths. 

    If given normalizor, output score in [0,1] and 1 denote 0 average weight, which is the ideal situation, 
    0 denote the average weight equals to maximal weight

    Parameters
    ----------
    selected_paths
        List store the path index picked in order
    paths
        The output of shortest_path algorithm, dictionary including all-pair shortest path
    adj_matrix
        adjacency matrix of the graph
    paths_num
        number of path printed
    Normalizor
        numeric value for score normalization
    Returns
    -------
    None
    """
    if paths_num == None:
        paths_num = len(selected_paths)
    for i in range(paths_num):
        path_now = paths[selected_paths[i]]
        
        weights = 0
        for count, index in enumerate(path_now):
            if count == 0:
                pre_idx = path_now[0] 
                weights = 0
            else:
                weights = weights + adj_matrix[pre_idx,index]
                pre_idx = index
        ave_weights = weights/len(path_now)
        if normalizor != None:
            print("path:",i, "number of meta cells in path:", len(path_now), "score(/1):", 1 - ave_weightsrmalizor, "(1 is ideal with 0 average weight)")
        else:
            print("path:",i, "number of meta cells in path:", len(path_now), "average weight:", ave_weights)