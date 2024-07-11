import itertools as it
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import scipy as sp
import torch
from entitybert import selection
from entitybert.selection import (
    EntityDto,
    EntityGraph,
    EntityTree,
    FileGraph,
    is_filename_valid,
)
from sentence_transformers import SentenceTransformer, losses
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)

_eucledian_dist = losses.BatchHardTripletLossDistanceFunction.eucledian_distance  #  type: ignore


def is_class(entity: EntityDto) -> bool:
    return entity.kind == "Class"


def is_attribute(entity: EntityDto) -> bool:
    return entity.kind == "Field"


def is_method(entity: EntityDto) -> bool:
    return entity.kind == "Constructor" or entity.kind == "Method"


class MethodAttributeGraph:
    def __init__(self, entities: list[EntityDto], subgraph: EntityGraph):
        self._entities: dict[str, EntityDto] = {e.id: e for e in entities}
        self._ids = set(self._entities.keys())
        self._subgraph = subgraph
        if len(self._entities) != len(entities):
            raise RuntimeError("entities are not unique")

    @cache
    def attributes(self) -> list[EntityDto]:
        return [e for e in self._entities.values() if is_attribute(e)]

    @cache
    def methods(self) -> list[EntityDto]:
        return [e for e in self._entities.values() if is_method(e)]

    @cache
    def attributes_for(self, id: str) -> list[EntityDto]:
        outgoing_ids = self._subgraph.only_outgoing(id, self._ids)
        outgoing_entities = (self._entities[id] for id in outgoing_ids)
        return [e for e in outgoing_entities if is_attribute(e)]

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


def calc_model_based_cohesion(embeddings: Tensor, kernel: np.ndarray) -> ModelBasedCohesion:
    # Without structure
    dists = calc_dists_to_center(embeddings)
    avg_dist_to_center = dists.mean().cpu().item()
    max_dist_to_center = dists.max().cpu().item()
    sum_dist_to_center = dists.sum().cpu().item()

    # With structure
    kernel = kernel.astype(np.float32) # mps issue
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


def calc_measure_df(db_path: str, model: SentenceTransformer) -> pd.DataFrame | None:
    try:
        conn = selection.open_db(db_path)
        file_graph = FileGraph.load_from_db(conn.cursor())
        trees = EntityTree.load_from_db(conn.cursor())
        graph = EntityGraph.load_from_db(conn.cursor())
    except Exception:
        return None
    rows: list[dict[str, Any]] = []
    for filename, tree in tqdm(trees.items()):
        if not is_filename_valid(filename):
            continue
        groups = tree.nontrivial_leaf_siblings()
        if len(groups) != 1:
            continue
        siblings = groups[0]
        entities = [tree[id] for id in siblings]
        parent_id = entities[0].parent_id
        if parent_id is None or not is_class(tree[parent_id]):
            continue
        parent = tree[parent_id]
        texts = [tree.entity_text(id) for id in siblings]
        row: dict[str, Any] = dict()
        row["DB"] = db_path
        row["Filename"] = filename
        row["Name"] = parent.name
        row["Kind"] = parent.kind
        row["LOC"] = tree.loc()
        row["Entities"] = len(entities)

        # LCOM1, LCOM2, LCOM3
        subgraph = graph.subgraph(set(siblings))
        mag = MethodAttributeGraph(entities, subgraph)
        mcg = MethodCouplingGraph(mag)
        row["Fields"] = len(mag.attributes())
        row["Methods"] = len(mag.methods())
        row["LCOM1"] = mcg.lcom1()
        row["LCOM2"] = mag.lcom2()
        row["LCOM3"] = mag.lcom3()

        # Model-based Semantic Cohesion
        embeddings = calc_embeddings(model, texts)
        kernel = calc_commute_time_kernel(siblings, subgraph.to_pairs())
        msc = calc_model_based_cohesion(embeddings, kernel)
        row["MSC_avg"] = msc.avg_dist_to_center
        row["MSC_max"] = msc.max_dist_to_center
        row["MSC_sum"] = msc.sum_dist_to_center
        row["MSC_sumsq"] = msc.sum_dist_to_center ** 2
        row["MSC_avg_s"] = msc.avg_dist_to_center_s
        row["MSC_max_s"] = msc.max_dist_to_center_s
        row["MSC_sum_s"] = msc.sum_dist_to_center_s
        row["MSC_sumsq_s"] = msc.sum_dist_to_center ** 2

        # Weird LCOM1 hybrid
        sims = dict()
        dists = _eucledian_dist(embeddings).cpu().numpy()
        for i, i_id in enumerate(siblings):
            for j, j_id in enumerate(siblings):
                sims[(i_id, j_id)] = 1 / (1 + dists[i][j])
        row["MSC_LCOM1"] = mcg.lcom1_sims(sims)
        pairwise_sum = 0
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                pairwise_sum += dists[i][j]
        row["MSC_sum_pairwise"] = pairwise_sum

        rows.append(row)
    return pd.DataFrame.from_records(rows)  # type: ignore


def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c, d in zip(df.columns, df.dtypes) if np.issubdtype(d, np.number)]


def calc_db_level_coefs(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for db, group_df in df.groupby("DB"):
        for a, b in it.combinations(get_numeric_cols(df), r=2):
            x, y = list(group_df[a]), list(group_df[b])
            tau = sp.stats.kendalltau(x, y, variant="c").statistic
            row1 = {"DB": db, "A": a, "B": b}
            row1["Tau"] = tau
            row2 = row1.copy()
            row2["A"], row2["B"] = b, a
            rows.extend([row1, row2])
    res_df = pd.DataFrame.from_records(rows)
    return res_df.sort_values(["DB", "A", "B"])


def calc_all_cohesion_df(
    db_paths: list[str], model_path: str, *, pbar: bool = False
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    skipped_paths: list[str] = []
    model = SentenceTransformer(model_path)
    for db_path in tqdm(db_paths, disable=not pbar):
        cohesion_df = calc_measure_df(db_path, model)
        if cohesion_df is not None:
            dfs.append(cohesion_df)
        else:
            skipped_paths.append(db_path)
    for path in skipped_paths:
        logger.warn(f"Could not load file data from {path}")
    return pd.concat(dfs, ignore_index=True)  # type: ignore
