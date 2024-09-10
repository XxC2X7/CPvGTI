import numpy as np  
import pandas as pd
import scvelo as scv
import scipy.sparse as sparse
import Clustering as clust
import NN as nn
import Path as path

def pairwise_distances(x, y):
    """\
    Description
        Calculate the pairwise distance given two feature matrices x and y
    Parameters
    ----------
    x
        feature matrix of dimension (n_obs_x, n_features)
    y
        feature matrix of dimension (n_obs_y, n_features)
    """
    x_norm = np.sum(x**2, axis = 1)[:, None]
    y_norm = np.sum(y**2, axis = 1)[None, :]
    
    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return np.where(dist <= 0.0, 0.0, dist)


class CPvGTI():
    def __init__(self, adata, preprocess = True, **kwargs):#初始化，数据预处理
        """\
        Description
            Initialize CPvGTI object
        
        Parameters
        ----------       
        adata
            Anndata object, store the dataset
        preprocess
            dataset is processed or not, boolean
        **kwargs
            additional parameters for preprocessing (using scvelo)
        """
        self.adata = adata
        if not sparse.issparse(adata.X):
            self.adata.X = sparse.csr_matrix(self.adata.X)
        
        if preprocess != True:
            print("preprocessing data using scvelo...")

            _kwargs = {
                "min_shared_genes": 20,
                "n_top_genes": 1000,
                "n_pcs": 30,
                "n_neighbors": 30,
                "velo_mode": "stochastic"
            }
            
            _kwargs.update(kwargs)

            # store the raw count before processing, for subsequent de analysis
            self.adata.layers["raw"] = self.adata.X.copy()

            scv.pp.filter_and_normalize(data = self.adata, 
                                        min_shared_counts=_kwargs["min_shared_genes"], 
                                        n_top_genes=_kwargs["n_top_genes"])

            scv.pp.moments(data = self.adata, n_pcs = _kwargs["n_pcs"], n_neighbors = _kwargs["n_neighbors"])

            if _kwargs["velo_mode"] == "stochastic":
                scv.tl.velocity(data = self.adata, model = _kwargs["velo_mode"])

            elif _kwargs["velo_mode"] == "dynamical":
                scv.tl.recover_dynamics(data = self.adata)
                scv.tl.velocity(data = self.adata, model = _kwargs["velo_mode"])

                # remove nan genes
                _velo_matrix = self.adata.layers['velocity'].copy()
                _genes_subset = ~np.isnan(_velo_matrix).any(axis=0)
                self.adata._inplace_subset_var(_genes_subset)

            else:
                raise ValueError("`velo_mode` can only be dynamical or stochastic")
                    
    def cp_construction(self, flavor = "gmm", n_clusters = None, resolution = 30, **kwarg):
        """\
        Description
            Constructing cellpopulation

        Parameters
        ----------
        flavor
            The clustering algorithm for cellpopulation: including gmm and leiden algorithm
        n_clusters
            The number of cellpopulation (if use gmm)
        resolution
            The resolution parameter(if use leiden), default 30
        """
        _kwargs = {
            # Boolean, whether include unspliced count or not
            "include_unspliced":True,
            # Standardize before pca, boolean
            "standardize": True,
            "n_comps": 30, 
            "kernel": "rbf",
            "alpha": 1,
            "length_scale": 0.3,
            "verbose": True,
            "seed": 0
        }
        _kwargs.update(kwarg)

        # skip if already clustered
        self.groups = clust.cluster_cells(self.adata, n_clusters = n_clusters,
                                          n_comps = _kwargs["n_comps"], resolution = resolution,
                                          include_unspliced = _kwargs["include_unspliced"],
                                          standardize = _kwargs["standardize"], seed = _kwargs["seed"], 
                                          flavor = flavor)

        # checked
        self.X_clust, self.velo_clust= clust.cellpopulation(self.adata, kernel = _kwargs["kernel"], 
                                      alpha = _kwargs["alpha"], length_scale = _kwargs["length_scale"])

        if _kwargs["verbose"] == True:
            print("CellPopulation constructed, number of cps: {:d}".format(len(self.X_clust)))
    
    def cp_graph(self, k_neighs = 10, pruning = True, **kwargs):
        """\
        Description
            cellpopulation level graph construction

        Parameters
        ----------
        k_neighs
            Number of neighbors for the neighborhood graph, default 10
        pruning
            Pruning the network or not, boolean variable, affect the continuity of the trajectory. 
            False (less fragmented paths) for most of the cases.
        """        
        _kwargs = {
            "symm": True,
            "scaling": 3,
            "distance_scalar": 0.5,
            "verbose": True
        }
        _kwargs.update(kwargs)

        _adj_matrix, _dist_matrix = nn.NeighborhoodGraph(self.X_clust, k_neighs = k_neighs, symm = _kwargs["symm"], pruning = pruning)

        self.adj_assigned = nn._assign_weights(connectivities = _adj_matrix, distances = _dist_matrix, X = self.X_clust, 
                                                               velo_pca = self.velo_clust, scaling = _kwargs["scaling"], 
                                                               distance_scalar = _kwargs["distance_scalar"], threshold = 0.0)

        self.max_weight = (_kwargs["scaling"] * (1 + _kwargs["distance_scalar"]))**_kwargs["scaling"]
        if _kwargs["verbose"] == True:
            print("CellPopulation level neighborhood graph constructed")
    
    def cp_paths_finding(self, threshold = 0.5, cutoff_length = 5, length_bias = 0.7,mode = "fast", **kwargs):
        """\
        Description
            CellPopulation level trajectory finding

        Parameters
        ----------
        threshold
            Cut-off quality score, equals to threshold * max_weight
        cutoff_length
            The cutoff length (lower bound) of inferred trajectory
        length_bias
            The bias on the path length for greedy selection
        mode
            The path finding algorithm. ``fast'': dijkstra, ``exact'': floydWarshall, default fast.
        """ 
        _kwargs = {
            "max_trajs": None,
            "verbose": True,
            "root_cell_indeg":[0,1,2],
        }

        _kwargs.update(kwargs)
        if mode == "fast":
            self.paths, self.opt = path.dijkstra_paths(adj = self.adj_assigned.copy(), indeg = _kwargs["root_cell_indeg"])
        elif mode == "exact":
            self.paths, self.opt = path.floyd_warshall(adj = self.adj_assigned.copy())
        else:
            raise ValueError("mode can only be ``fast'' or ``exact''.")
        n_cps = int((np.max(self.groups)+1))
        self.greedy_order, self.paths = path.g_selection(nodes = n_cps, paths = self.paths,opt_value = self.opt, threshold = threshold, 
                                                              max_w=self.max_weight, cut_off=cutoff_length, 
                                                              verbose = _kwargs["verbose"], length_bias = length_bias, 
                                                              max_trajs = _kwargs["max_trajs"])

    def _cells_insertion(self, num_trajs = None, prop_insert = 1):
        """\
        Description
            Inserting unassigned cells

        Parameters
        ----------
        num_trajs
            Number of trajectories
        verbose
            Output result
        prop_insert
            Parameters for the number of cells incorporated
        """     
        if num_trajs == None:
            num_trajs = len(self.greedy_order)
        elif num_trajs > len(self.greedy_order):
            print(len(self.greedy_order))

        # assigned cells
        cell_pools = set([])
        clust_pools = set([])

        for i in range(num_trajs):
            traj = []
            for index in self.paths[self.greedy_order[i]]: 
                clust_pools.add(index)  

                # find the cells corresponding to the cluster in greedy paths
                group_i = np.where(self.groups == index)[0]
                # ordering the cells
                diff = self.adata.obsm['X_pca'][group_i,:] - self.X_clust[index,:] 
                group_i = group_i[np.argsort(np.dot(diff, self.velo_clust[index,:])/np.linalg.norm(self.velo_clust[index,:],2))]

                # traj store all the cells in the trajectory/greedy paths
                traj = np.append(traj, group_i)

            # incorporate all the cells in traj
            cell_pools = cell_pools.union(set(traj))
        
        # uncovered cells
        cell_uncovered = list(set([x for x in range(self.adata.n_obs)]) - cell_pools)

        # cell_uncovered by meta-cell distance matrix calculation
        X_pca = self.adata.obsm["X_pca"]
        uncovered_pca = X_pca[cell_uncovered, :]

        covered_clust = self.X_clust[list(clust_pools), :]
        pdist = pairwise_distances(uncovered_pca, covered_clust)

        threshold = prop_insert * np.max(pdist)

        pdist = np.where(pdist <= threshold, pdist, np.inf)
        

        # choose cellpopulation for each cell
        # meta_cells = [list(clust_pools)[x] for i, x in enumerate(np.argmin(pdist, axis = 1)) if pdist[i, x] != np.inf]
        
        cps = []
        indices = []

        for idx in range(pdist.shape[0]):
            x = np.argmin(pdist[idx,:])
            if pdist[idx, x] != np.inf:
                cps.append(list(clust_pools)[x])
                indices.append(cell_uncovered[idx])

        print("number of cells: " + str(len(indices)))
        # assign cells to the closest cp
        self.groups[indices] = cellpopulation


    def first_order_pt(self, num_trajs = None, verbose = True, prop_insert = 0):
        """\
        Description
            cell level pseudo-time inference using first order approximation

        Parameters
        ----------
        num_trajs
            Number of trajectories
        verbose
            Output result
        prop_insert
            The proportion of cells to be incorporated, default 0(no cell to be inserted)
        """ 
        if num_trajs == None:
            num_trajs = len(self.greedy_order)
        elif num_trajs > len(self.greedy_order):
            print(len(self.greedy_order))
            num_trajs = len(self.greedy_order)
            # raise ValueError("number of trajectory to be selected larger than maximum number")
        self.pseudo_order = pd.DataFrame(data = np.nan, index = self.adata.obs.index, columns = ["traj_" + str(x) for x in range(num_trajs)]) 

        if prop_insert > 0:
            # inserting uncovered cells to the nearby cps
            self._cells_insertion(num_trajs = num_trajs, prop_insert = prop_insert)


        for i in range(num_trajs):
            traj = np.array([])

            # the traj is already sorted by enumerating process
            for index in self.paths[self.greedy_order[i]]:
                # find the cells corresponding to the cluster in greedy paths
                group_i = np.where(self.groups == index)[0]
                # ordering the cells
                diff = self.adata.obsm['X_pca'][group_i,:] - self.X_clust[index,:] 
                group_i = group_i[np.argsort(np.dot(diff, self.velo_clust[index,:])/np.linalg.norm(self.velo_clust[index,:],2))]

                # traj store all the cells in the trajectory/greedy paths
                traj = np.append(traj, group_i)  
                
            traj = traj.astype('int')

            for pt, curr_cell in enumerate(traj):
                # self.pseudo_order.loc["cell_"+str(curr_cell),"traj_"+str(i)] = pt
                
                self.pseudo_order.loc[self.adata.obs.index.values[curr_cell],"traj_"+str(i)] = pt

        if verbose:
            print("Cell-level pseudo-time inferred")
        
    def compute_transitions(self, density_normalize=True):
        T = self._adata.uns["velocity_graph"] - self._adata.uns["velocity_graph_neg"]
        self._connectivities = T
        if density_normalize:
            q = np.asarray(T.sum(axis=0))
            q += q == 0
            Q = (
                spdiags(1.0 / q, 0, T.shape[0], T.shape[0])
                if issparse(T)
                else np.diag(1.0 / q)
            )  
            K = Q.dot(T).dot(Q)
        else:
            K = T
        z = np.sqrt(np.asarray(K.sum(axis=0)))
        Z = (
            spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
            if issparse(K)
            else np.diag(1.0 / z)
        )
        self._transitions_sym = Z.dot(K).dot(Z)   
    def compute_eigen(self, n_comps=10, sym=None, sort="decrease"):
        if self._transitions_sym is None:
            raise ValueError("Run `.compute_transitions` first.")
        n_comps = min(self._transitions_sym.shape[0] - 1, n_comps)
        evals, evecs = linalg.eigsh(self._transitions_sym, k=n_comps, which="LM")
        self._eigen_values = evals[::-1]
        self._eigen_basis = evecs[:, ::-1]
    
    def compute_pseudotime(self, inverse=False):
        if self.iroot is not None:
            self._set_pseudotime()
            self.pseudotime = 1 - self.pseudotime if inverse else self.pseudotime
            self.pseudotime[~np.isfinite(self.pseudotime)] = np.nan
        else:
            self.pseudotime = np.empty(self._adata.n_obs)
            self.pseudotime[:] = np.nan
