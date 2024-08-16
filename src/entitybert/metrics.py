import itertools as it
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import scipy as sp
import torch
import entitybert.lcsm
from entitybert.embeddings import Embedder, load_caching_embedder
from entitybert.selection import (
    EntityDto,
    EntityGraph,
    EntityTree,
    calc_canonical,
    iter_standard_classes,
)
from ordered_set import OrderedSet as oset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, floyd_warshall
from sentence_transformers import SentenceTransformer, losses
from torch import Tensor

_eucledian_dist = losses.BatchHardTripletLossDistanceFunction.eucledian_distance  #  type: ignore


def calc_commute_time_kernel(sym_adj_mat: np.ndarray) -> np.ndarray:
    degree_mat = np.diag(sym_adj_mat.sum(axis=1))
    laplacian_mat = degree_mat - sym_adj_mat
    laplacian_pseudo_inverse = np.linalg.pinv(laplacian_mat)
    return laplacian_pseudo_inverse


@dataclass
class ModelBasedCohesion:
    avg_dist_to_center: float
    max_dist_to_center: float
    sum_dist_to_center: float
    avg_dist_to_center_s: float
    max_dist_to_center_s: float
    sum_dist_to_center_s: float


def calc_dists_to_center(embeddings: Tensor) -> Tensor:
    center = torch.mean(embeddings, dim=0).unsqueeze(0)  # type: ignore
    dists = _eucledian_dist(torch.vstack((embeddings, center)))  # type: ignore
    return dists[-1][:-1]


def calc_model_based_cohesion(
    embeddings: Tensor, kernel: np.ndarray
) -> ModelBasedCohesion:
    # Without structure
    dists = calc_dists_to_center(embeddings)
    avg_dist_to_center = dists.mean().cpu().item()
    max_dist_to_center = dists.max().cpu().item()
    sum_dist_to_center = dists.sum().cpu().item()

    # With structure
    kernel = kernel.astype(np.float32)  # mps issue
    kernel_torch = torch.from_numpy(kernel).to(embeddings.device)
    enhanced_embeddings = torch.matmul(kernel_torch, embeddings)
    dists = calc_dists_to_center(enhanced_embeddings)
    avg_dist_to_center_s = dists.mean().cpu().item()
    max_dist_to_center_s = dists.max().cpu().item()
    sum_dist_to_center_s = dists.sum().cpu().item()

    return ModelBasedCohesion(
        avg_dist_to_center,
        max_dist_to_center,
        sum_dist_to_center,
        avg_dist_to_center_s,
        max_dist_to_center_s,
        sum_dist_to_center_s,
    )


def calc_metrics_row(
    tree: EntityTree, subgraph: EntityGraph, embedder: Embedder
) -> dict[str, Any]:
    row: dict[str, Any] = dict()

    # "Members" are the methods and attributes of the standard class
    row["Members"] = len(subgraph.nodes)
    row["Methods"] = len(subgraph.nodes.methods())
    row["Fields"] = len(subgraph.nodes.attributes())

    # Canonical metrics
    canon = calc_canonical(subgraph)
    row["LCOM1"] = canon.lcom1
    row["LCOM2"] = canon.lcom2
    row["LCOM3"] = canon.lcom3
    row["LCOM4"] = canon.lcom4
    row["Co"] = canon.co
    row["TCC"] = canon.tcc
    row["LCC"] = canon.lcc
    row["LCOM5"] = canon.lcom5

    # Marcus & Poshyvanyk
    docs = entitybert.lcsm.find_documents(tree.text())
    if len(docs) == 0:
        row["NC3"] = None
        row["LCSM"] = None
    else:
        sim_mat = entitybert.lcsm.calc_sim_mat(docs)
        acsm = entitybert.lcsm.calc_acsm(sim_mat)
        c3 = entitybert.lcsm.calc_c3(acsm)
        lcsm = entitybert.lcsm.calc_lcsm(sim_mat, acsm)
        row["NC3"] = -1 * c3
        row["LCSM"] = lcsm

    # Model-based Semantic Cohesion
    texts = [tree.entity_text(m.id) for m in subgraph.nodes]
    embeddings_dict = embedder.embed(texts, pbar=False)
    embeddings_arr = np.array([embeddings_dict[t] for t in texts])
    embeddings = torch.Tensor(embeddings_arr).to(embedder.device())
    kernel = calc_commute_time_kernel(subgraph.to_sym_adj_mat())
    msc = calc_model_based_cohesion(embeddings, kernel)
    row["MSC_avg"] = msc.avg_dist_to_center
    row["MSC_max"] = msc.max_dist_to_center
    row["MSC_sum"] = msc.sum_dist_to_center
    row["MSC_sumsq"] = msc.sum_dist_to_center**2
    row["MSC_avg_s"] = msc.avg_dist_to_center_s
    row["MSC_max_s"] = msc.max_dist_to_center_s
    row["MSC_sum_s"] = msc.sum_dist_to_center_s
    row["MSC_sumsq_s"] = msc.sum_dist_to_center**2

    return row


def calc_metrics_df(
    files_df: pd.DataFrame, embedder: Embedder, *, pbar: bool
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for input_row, tree, subgraph in iter_standard_classes(files_df, pbar=pbar):
        row = input_row.to_dict()
        metrics_row = calc_metrics_row(tree, subgraph, embedder)
        row.update(metrics_row)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c, d in zip(df.columns, df.dtypes) if np.issubdtype(d, np.number)]


def calc_db_level_coefs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for db, group_df in metrics_df.groupby("db_path"):
        for a, b in it.combinations(get_numeric_cols(metrics_df), r=2):
            x, y = list(group_df[a]), list(group_df[b])
            tau = sp.stats.kendalltau(x, y, variant="c").statistic
            row1 = {"db_path": db, "A": a, "B": b}
            row1["Tau"] = tau
            row2 = row1.copy()
            row2["A"], row2["B"] = b, a
            rows.extend([row1, row2])
    res_df = pd.DataFrame.from_records(rows)
    return res_df.sort_values(["db_path", "A", "B"])
