import numpy as np
import anndata
import pandas as pd


def kendalltau(pt_pred, pt_true):
    """\
    Description
        kendall tau correlationship
    
    Parameters
    ----------
    pt_pred
        inferred pseudo-time
    pt_true
        ground truth pseudo-time
    Returns
    -------
    tau
        returned score
    """
    from scipy.stats import kendalltau
    pt_true = pt_true.squeeze()
    pt_pred = pt_pred.squeeze()
    tau, p_val = kendalltau(pt_pred, pt_true)
    return tau


def CPvGTI_kt(CPvGTI_obj):
    """\
    Description
        kendall tau correlationship for CPvGTI
    
    Parameters
    ----------
    CPvGTI_obj
        CPvGTI object
    Returns
    -------
    kt
        returned score    
    """
    if "sim_time" not in CPvGTI_obj.adata.obs.columns:
        raise ValueError("ground truth value not provided")

    pseudo_order = CPvGTI_obj.pseudo_order
    non_zeros = {}
    pt_pred = {}
    pt_true = {}
    kt = {}
    for icol, col in enumerate(pseudo_order.columns):
        non_zeros[col] = np.where(~np.isnan(pseudo_order[col].values.squeeze()))[0]
        pt_pred[col] = pseudo_order.iloc[non_zeros[col], icol].values.squeeze()
        pt_true[col] = CPvGTI_obj.adata.obs["sim_time"].iloc[non_zeros[col]].values
        kt[col] = kendalltau(pt_pred[col], pt_true[col])
    return kt

from scipy.stats import spearmanr
import numpy as np

def spearmanr_corr(pt_pred, pt_true):
    """\
    Description
        Spearman's rank correlation coefficient
    
    Parameters
    ----------
    pt_pred
        inferred pseudo-time
    pt_true
        ground truth pseudo-time
    Returns
    -------
    rho
        Spearman correlation coefficient
    """
    pt_true = pt_true.squeeze()
    pt_pred = pt_pred.squeeze()
    rho, p_val = spearmanr(pt_pred, pt_true)
    return rho

def CPvGTI_spearman(CPvGTI_obj):
    """\
    Description
        Spearman's rank correlation coefficient for CPvGTI
    
    Parameters
    ----------
    CPvGTI_obj
        CPvGTI object
    Returns
    -------
    sr
        returned score    
    """
    if "sim_time" not in CPvGTI_obj.adata.obs.columns:
        raise ValueError("ground truth value not provided")

    pseudo_order = CPvGTI_obj.pseudo_order
    non_zeros = {}
    pt_pred = {}
    pt_true = {}
    sr = {}
    for icol, col in enumerate(pseudo_order.columns):
        non_zeros[col] = np.where(~np.isnan(pseudo_order[col].values.squeeze()))[0]
        pt_pred[col] = pseudo_order.iloc[non_zeros[col], icol].values.squeeze()
        pt_true[col] = CPvGTI_obj.adata.obs["sim_time"].iloc[non_zeros[col]].values
        sr[col] = spearmanr(pt_pred[col], pt_true[col])[0]  # Only get the correlation coefficient
    return sr