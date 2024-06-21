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


@dataclass
class ModelBasedCohesion:
    avg_dist_to_center: float
    max_dist_to_center: float
    total_variance: float
    spectral_norm: float


def calc_model_based_cohesion(
    model: SentenceTransformer, texts: list[str]
) -> ModelBasedCohesion:
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    center = torch.mean(embeddings, dim=0).unsqueeze(0)  # type: ignore
    dists = _eucledian_dist(torch.vstack((embeddings, center)))  # type: ignore
    dists_to_mean = dists[-1][0:-1]
    avg_dist_to_center = dists_to_mean.mean().cpu().item()
    max_dist_to_center = dists_to_mean.max().cpu().item()
    cov = np.cov(embeddings.cpu().numpy().T)  # type: ignore
    total_variance = np.trace(cov)
    spectral_norm = sp.linalg.eigh(
        cov, eigvals_only=True, subset_by_index=[len(cov) - 1, len(cov) - 1]
    ).item()
    return ModelBasedCohesion(
        avg_dist_to_center, max_dist_to_center, total_variance, spectral_norm
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
        row["[E] LOC"] = tree.loc()
        row["[E] Entities"] = len(entities)

        # LCOM1, LCOM2, LCOM3
        subgraph = graph.subgraph(set(siblings))
        mag = MethodAttributeGraph(entities, subgraph)
        mcg = MethodCouplingGraph(mag)
        row["[E] Fields"] = len(mag.attributes())
        row["[E] Methods"] = len(mag.methods())
        row["[C] LCOM1"] = mcg.lcom1()
        row["[C] LCOM2"] = mag.lcom2()
        row["[C] LCOM3"] = mag.lcom3()

        # Model-based Semantic Cohesion
        msc = calc_model_based_cohesion(model, texts)
        row["[C] MSC1"] = msc.avg_dist_to_center
        row["[C] MSC2"] = msc.max_dist_to_center
        row["[C] MSC3"] = msc.total_variance
        row["[C] MSC4"] = msc.spectral_norm

        # History scores
        cochanges = file_graph.cochange_counts(filename)
        nontrivial_cochanges = file_graph.nontrivial_cochange_counts(filename)
        n_changes = len(file_graph.commits_of(filename))
        row["[H] Changes"] = n_changes
        row["[H] Cochanges"] = cochanges.total()
        row["[H] Cochanges (>1)"] = nontrivial_cochanges.total()
        row["[H] Unique Cochanges"] = len(cochanges)
        row["[H] Unique Cochanges (>1)"] = len(nontrivial_cochanges)
        row["[H] Cochange Rate"] = row["[H] Cochanges"] / n_changes
        row["[H] Cochange (>1) Rate"] = row["[H] Cochanges (>1)"] / n_changes
        row["[H] Unique Cochange Rate"] = row["[H] Unique Cochanges"] / n_changes
        row["[H] Unique Cochange (>1) Rate"] = row["[H] Unique Cochanges (>1)"] / n_changes

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
