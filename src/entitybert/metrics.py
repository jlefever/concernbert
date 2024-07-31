import itertools as it
import logging
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import scipy as sp
import torch
from entitybert.selection import (
    EntityDto,
    EntityGraph,
    iter_entity_graphs,
)
from sentence_transformers import SentenceTransformer, losses
from torch import Tensor

logger = logging.getLogger(__name__)

_eucledian_dist = losses.BatchHardTripletLossDistanceFunction.eucledian_distance  #  type: ignore


class MethodAttributeGraph:
    def __init__(self, members: list[EntityDto], subgraph: EntityGraph):
        self._members: dict[str, EntityDto] = {e.id: e for e in members}
        self._ids = set(self._members.keys())
        self._subgraph = subgraph
        if len(self._members) != len(members):
            raise RuntimeError("members are not unique")

    @cache
    def attributes(self) -> list[EntityDto]:
        return [e for e in self._members.values() if e.is_attribute()]

    @cache
    def methods(self) -> list[EntityDto]:
        return [e for e in self._members.values() if e.is_method()]

    @cache
    def attributes_for(self, id: str) -> list[EntityDto]:
        outgoing_ids = self._subgraph.only_outgoing(id, self._ids)
        outgoing_members = (self._members[id] for id in outgoing_ids)
        return [e for e in outgoing_members if e.is_attribute()]

    @cache
    def edges(self) -> set[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()
        for method in self.methods():
            for attribute in self.attributes_for(method.id):
                edges.add((method.id, attribute.id))
        return edges

    @cache
    def lcom2(self) -> float | None:
        n = len(self.methods())
        a = len(self.attributes())
        e = len(self.edges())
        if n == 0 or a == 0:
            return None
        return 1 - (e / (n * a))

    @cache
    def lcom3(self) -> float | None:
        n = len(self.methods())
        if n < 2:
            return None
        lcom2 = self.lcom2()
        if lcom2 is None:
            return None
        return (n / (n - 1)) * lcom2


class MethodCouplingGraph:
    def __init__(self, mag: MethodAttributeGraph):
        self._mag = mag

    def n_nodes(self) -> int:
        return len(self._mag.methods())

    def n_edges(self) -> int:
        n_edges = 0
        methods = list(self._mag.methods())
        for i in range(len(methods)):
            i_id = methods[i].id
            i_attributes = self._mag.attributes_for(i_id)
            i_attributes = set(a.id for a in i_attributes)
            for j in range(i + 1, len(methods)):
                j_id = methods[j].id
                j_attributes = self._mag.attributes_for(j_id)
                j_attributes = set(a.id for a in j_attributes)
                if len(i_attributes & j_attributes) != 0:
                    n_edges += 1
        return n_edges

    def lcom1(self) -> float:
        n = self.n_nodes()
        e = self.n_edges()
        return (0.5 * (n * (n - 1))) - e

    def lcom1_sims(self, sims: dict[tuple[str, str], float]) -> float:
        total = 0
        methods = list(self._mag.methods())
        for i in range(len(methods)):
            i_id = methods[i].id
            i_attributes = self._mag.attributes_for(i_id)
            i_attributes = set(a.id for a in i_attributes)
            for j in range(i + 1, len(methods)):
                j_id = methods[j].id
                j_attributes = self._mag.attributes_for(j_id)
                j_attributes = set(a.id for a in j_attributes)
                if len(i_attributes & j_attributes) == 0:
                    total += sims[(i_id, j_id)]
        return total


def calc_adjacency_matrix(entity_ids, edges):
    n = len(entity_ids)
    node_index = {node: i for i, node in enumerate(entity_ids)}
    adj_matrix = np.zeros((n, n))
    for edge in edges:
        i, j = node_index[edge[0]], node_index[edge[1]]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix


def calc_commute_time_kernel(entity_ids, edges):
    adj_matrix = calc_adjacency_matrix(entity_ids, edges)
    degree_matrix = np.diag(adj_matrix.sum(axis=1))
    laplacian_matrix = degree_matrix - adj_matrix
    laplacian_pseudo_inverse = np.linalg.pinv(laplacian_matrix)
    return laplacian_pseudo_inverse


def calc_embeddings(model: SentenceTransformer, texts: list[str]) -> Tensor:
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)  # type: ignore


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


def calc_metrics_df(
    files_df: pd.DataFrame, model: SentenceTransformer, *, pbar: bool
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for input_row, tree, graph in iter_entity_graphs(files_df, pbar=pbar):
        row = input_row.to_dict()
        rows.append(row)

        # Only consider "standard classes"
        cls = tree.standard_class()
        row["is_std_class"] = cls is not None
        if cls is None:
            continue

        # "Members" are the methods and attributes of the standard class
        members = tree.children(cls.id)
        member_ids = [m.id for m in members]
        row["members"] = len(members)

        # The "graph" object is the graph for the whole project
        # Use "subgraph" to get the the graph for the standard file
        subgraph = graph.subgraph(set(member_ids))

        # LCOM1, LCOM2, LCOM3
        mag = MethodAttributeGraph(members, subgraph)
        mcg = MethodCouplingGraph(mag)
        row["Fields"] = len(mag.attributes())
        row["Methods"] = len(mag.methods())
        row["LCOM1"] = mcg.lcom1()
        row["LCOM2"] = mag.lcom2()
        row["LCOM3"] = mag.lcom3()

        # Model-based Semantic Cohesion
        texts = [tree.entity_text(id) for id in member_ids]
        embeddings = calc_embeddings(model, texts)
        kernel = calc_commute_time_kernel(member_ids, subgraph.to_pairs())
        msc = calc_model_based_cohesion(embeddings, kernel)
        row["MSC_avg"] = msc.avg_dist_to_center
        row["MSC_max"] = msc.max_dist_to_center
        row["MSC_sum"] = msc.sum_dist_to_center
        row["MSC_sumsq"] = msc.sum_dist_to_center**2
        row["MSC_avg_s"] = msc.avg_dist_to_center_s
        row["MSC_max_s"] = msc.max_dist_to_center_s
        row["MSC_sum_s"] = msc.sum_dist_to_center_s
        row["MSC_sumsq_s"] = msc.sum_dist_to_center**2

        # Weird LCOM1 hybrid
        sims = dict()
        dists = _eucledian_dist(embeddings).cpu().numpy()
        for i, i_id in enumerate(member_ids):
            for j, j_id in enumerate(member_ids):
                sims[(i_id, j_id)] = 1 / (1 + dists[i][j])
        row["MSC_LCOM1"] = mcg.lcom1_sims(sims)
        pairwise_sum = 0
        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                pairwise_sum += dists[i][j]
        row["MSC_sum_pairwise"] = pairwise_sum

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
