import anndata
from scipy import sparse
import numpy as np  
import pandas as pd
import leiden
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def cluster_cells(
        adata, n_clusters = None, resolution = 30,
        n_comps = 30, include_unspliced = True, 
        standardize = True, seed = 0, flavor = "gmm"
        ):
    """\
    Cluster cells into clusters, using gmm

    Parameters
    ----------
    adata
        The annotated data matrix.
    n_clusters
        number of clusters, default, cell number/10
    include_unspliced
        Boolean, whether include unspliced count or not

    Returns
    -------
    `adata.obs[groups]`
        Array of dim (number of samples) that stores the subgroup id
        (`0`, `1`, ...) for each cell.
    """

    if 'groups' in adata.obs:
        print("Already conducted clustering")
        return adata.obs["groups"].values.astype("int64")
    else:
        if n_clusters == None:
            n_clusters = int(adata.n_obs/10)
        if sparse.issparse(adata.layers['spliced']):
            X_spliced = np.log1p(adata.layers['spliced'].toarray())
        else:
            X_spliced = np.log1p(adata.layers['spliced'])
        
        if sparse.issparse(adata.layers['unspliced']):
            X_unspliced = np.log1p(adata.layers['unspliced'].toarray())
        else:
            X_unspliced = np.log1p(adata.layers['unspliced'])

        if standardize:
            pca = Pipeline(
                [('scaling', StandardScaler(with_mean=True, with_std=True)), 
                ('pca', PCA(n_components=n_comps, svd_solver='arpack'))])
        else:
            pca = PCA(n_components=n_comps, svd_solver="arpack")

        gmm = GaussianMixture(n_components=n_clusters, random_state=seed)

        # Include unspliced count for clustering, recalculate X_pca and X_concat_pca
        if include_unspliced:
            
            X_concat = np.concatenate((X_spliced,X_unspliced),axis=1)
            X_concat_pca = pca.fit_transform(X_concat)
            X_pca = pca.fit_transform(X_spliced)
            if flavor == "gmm":
                print("using gmm")
                groups = gmm.fit_predict(X_concat_pca)  # 对拼接后的PCA结果进行聚类
            elif flavor == "leiden":
                print("using leiden")
                groups = leiden.cluster_cells_leiden(X = X_concat_pca, resolution = resolution, random_state = seed)
            elif flavor == "hier":
                print("using hier")
                groups = AgglomerativeClustering(n_clusters = n_clusters, affinity = "euclidean").fit(X_concat_pca).labels_            
            else:
                raise ValueError("flavor can only be `gmm', `leiden' or `hier'")
            
            adata.obsm['X_pca'] = X_pca
                
        else:
            X_pca = pca.fit_transform(X_spliced)
            adata.obsm['X_pca'] = X_pca
            
            if flavor == "gmm":
                print("using gmm")
                groups = gmm.fit_predict(X_pca)
            elif flavor == "leiden":
                print("using leiden")
                groups = leiden.cluster_cells_leiden(X = X_pca, resolution = resolution, random_state = seed)
            elif flavor == "hier":
                print("using hier")
                groups = AgglomerativeClustering(n_clusters = n_clusters, affinity = "euclidean").fit(X_pca).labels_            
            
            else:
                raise ValueError("flavor can only be `gmm', `leiden' or `hier'")

        velo_matrix = adata.layers["velocity"]
        # predict gene expression data
        X_pre = X_spliced + velo_matrix 
        # /np.linalg.norm(velo_matrix,axis=1)[:,None]
        adata.obsm['X_pre_pca'] = pca.transform(X_pre)
        adata.obsm['velocity_pca'] = adata.obsm['X_pre_pca'] - X_pca

        adata.obs['groups'] = groups.astype('int64')

        return groups.astype('int64')

def cell_population(adata, kernel = 'rbf', alpha = 0.3, length_scale = 0.3):
    """\
    estimate the expression and velocity of cellpopulation, using Gaussian Process Regression

    Parameters
    ----------
    adata
        The annotated data matrix.
    kernel
        kernel choice, default rbf kernel
    alpha 
        regularization
    length_scale 
        para of rbf kernel

    Returns
    -------
    X_cluster
        cellpopulation expression data
    vel_cluster
        cellpopualtion velocity data

    """
    
    if 'groups' not in adata.obs.columns or 'X_pca' not in adata.obsm:
        raise ValueError("please cluster cells first") 
    
    from sklearn.cluster import AgglomerativeClustering
    kernel_gp = RBF(length_scale)
    gp = GaussianProcessRegressor(kernel=kernel_gp, alpha=alpha)
      
    n_sub_clusters = 3
    groups = adata.obs['groups'].values
    n_clusters = int(np.max(groups) + 1)
    X_pca = adata.obsm['X_pca']
    X_clust = np.zeros((n_clusters,X_pca.shape[1]))
    velo_clust = np.zeros(X_clust.shape)
    X_cluster = {c: np.zeros((n_sub_clusters, X_pca.shape[1])) for c in range(n_clusters)}
    velo_cluster = {c: np.zeros_like(X_cluster[c]) for c in range(n_clusters)}
    velo_pca = adata.obsm['velocity_pca']
    
    for c in range(n_clusters):
        indices = np.where(groups == c)[0]
        gp.fit(X_pca[indices,:],velo_pca[indices,:])
        X_clust[c,:] = np.mean(X_pca[indices,:],axis=0)
        velo_clust[c,:] = gp.predict(X_clust[c,:][np.newaxis,:])
    
    sub_groups = {c: {} for c in range(n_clusters)}
    for c in range(n_clusters):
        indices = np.where(groups == c)[0]
        if len(indices) > n_sub_clusters:
            hierarchical = AgglomerativeClustering(n_clusters=n_sub_clusters, affinity='euclidean', linkage='ward')
            hierarchical.fit(X_pca[indices, :])
            sub_group_labels = hierarchical.labels_
        
            for sub_c in range(n_sub_clusters):
                sub_indices = indices[sub_group_labels == sub_c]
                sub_groups[c][sub_c] = sub_indices.tolist()
            
                gp.fit(X_pca[indices,:],velo_pca[indices,:])
                X_cluster[c][sub_c, :] = np.mean(X_pca[sub_indices, :], axis=0)
                velo_cluster[c][sub_c, :] = gp.predict(np.mean(X_pca[sub_indices, :], axis=0).reshape(1, -1))
            
        else:
            gp.fit(X_pca[indices,:], velo_pca[indices,:])
            X_cluster[c] = np.mean(X_pca[indices,:], axis=0)
            velo_cluster[c] = gp.predict(X_cluster[c][np.newaxis,:])
            sub_groups[c] = np.array([0] * len(indices))
        
    return X_clust,velo_clust,X_cluster, velo_cluster,sub_groups
    
def cellpopulation(adata, kernel = 'rbf', alpha = 0.3, length_scale = 0.3):
    """\
    estimate the expression and velocity of cellpopulation, using Gaussian Process Regression

    Parameters
    ----------
    adata
        The annotated data matrix.
    kernel
        kernel choice, default rbf kernel
    alpha 
        regularization
    length_scale 
        para of rbf kernel

    Returns
    -------
    X_cluster
        cellpopulation expression data
    vel_cluster
        cellpopualtion velocity data

    """
    
    if 'groups' not in adata.obs.columns or 'X_pca' not in adata.obsm:
        raise ValueError("please cluster cells first") 
    
    from sklearn.cluster import AgglomerativeClustering
    kernel_gp = RBF(length_scale)
    gp = GaussianProcessRegressor(kernel=kernel_gp, alpha=alpha)
      
    groups = adata.obs['groups'].values
    n_clusters = int(np.max(groups) + 1)
    X_pca = adata.obsm['X_pca']
    X_clust = np.zeros((n_clusters,X_pca.shape[1]))
    velo_clust = np.zeros(X_clust.shape)
    velo_pca = adata.obsm['velocity_pca']
    
    for c in range(n_clusters):
        indices = np.where(groups == c)[0]
        gp.fit(X_pca[indices,:],velo_pca[indices,:])
        X_clust[c,:] = np.mean(X_pca[indices,:],axis=0)
        velo_clust[c,:] = gp.predict(X_clust[c,:][np.newaxis,:])
        
    return X_clust,velo_clust