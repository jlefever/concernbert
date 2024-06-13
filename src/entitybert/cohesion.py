import logging
from functools import cache
from typing import Any

import pandas as pd
import torch
from entitybert import selection
from entitybert.selection import (
    EntityDto,
    EntityGraph,
    EntityTree,
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


def flatten_upper(mat) -> Tensor:
    values = []
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            values.append(mat[i][j])
    return torch.Tensor(values).to(mat.device)


def to_neighbors(mat):
    arr = []
    for i, row in enumerate(mat):
        arr.append(torch.cat([row[0:i], row[i + 1 :]]).unsqueeze(0))
    return torch.cat(arr)


def append_model_cohesion(
    row: dict[str, Any], model: SentenceTransformer, texts: list[str]
):
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    mean = torch.mean(embeddings, dim=0).unsqueeze(0) # type: ignore
    dists = _eucledian_dist(torch.vstack((embeddings, mean))) # type: ignore
    pairwise_dists = dists[0:-1, 0:-1]
    flat_pairwise_dists = flatten_upper(pairwise_dists)
    dists_to_mean = dists[-1][0:-1]
    neighbors = to_neighbors(pairwise_dists)
    near_neighbors, _ = torch.min(neighbors, dim=1)
    far_neighbors, _ = torch.max(neighbors, dim=1)
    row["AvgPD"] = flat_pairwise_dists.mean().cpu().item()
    row["MaxPD"] = flat_pairwise_dists.max().cpu().item()
    row["AvgDM"] = dists_to_mean.mean().cpu().item()
    row["MaxDM"] = dists_to_mean.max().cpu().item()
    row["AvgNN"] = near_neighbors.mean().cpu().item()
    row["MaxNN"] = near_neighbors.max().cpu().item()
    row["AvgFN"] = far_neighbors.mean().cpu().item()
    row["MaxFN"] = far_neighbors.max().cpu().item()


def calc_cohesion_df(db_path: str, model: SentenceTransformer) -> pd.DataFrame | None:
    try:
        conn = selection.open_db(db_path)
        trees = EntityTree.load_from_db(conn.cursor())
        graph = EntityGraph.load_from_db(conn.cursor())
    except Exception:
        return None
    rows: list[dict[str, Any]] = []
    for filename, tree in tqdm(trees.items()):
        if not is_filename_valid(filename):
            continue
        for i, siblings in enumerate(tree.nontrivial_leaf_siblings()):
            if len(siblings) == 1:
                print("????????")
            entities = [tree[id] for id in siblings]
            parent_id = entities[0].parent_id
            if parent_id is None or not is_class(tree[parent_id]):
                continue
            texts = [tree.entity_text(id) for id in siblings]
            parent = tree[parent_id]
            subgraph = graph.subgraph(set(siblings))
            mag = MethodAttributeGraph(entities, subgraph)
            mcg = MethodCouplingGraph(mag)
            row: dict[str, Any] = dict()
            row["DB"] = db_path
            row["Filename"] = filename
            row["Group"] = f"Group {i + 1}"
            row["Name"] = parent.name
            row["Kind"] = parent.kind
            row["LOC"] = parent.end_row - parent.start_row
            row["Entities"] = len(entities)
            row["Attributes"] = len(mag.attributes())
            row["Methods"] = len(mag.methods())
            row["LCOM1"] = mcg.lcom1()
            row["LCOM2"] = mag.lcom2()
            row["LCOM3"] = mag.lcom3()
            append_model_cohesion(row, model, texts)
            rows.append(row)
    return pd.DataFrame.from_records(rows)  # type: ignore


def calc_all_cohesion_df(
    db_paths: list[str], model_path: str, *, pbar: bool = False
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    skipped_paths: list[str] = []
    model = SentenceTransformer(model_path)
    for db_path in tqdm(db_paths, disable=not pbar):
        cohesion_df = calc_cohesion_df(db_path, model)
        if cohesion_df is not None:
            dfs.append(cohesion_df)
        else:
            skipped_paths.append(db_path)
    for path in skipped_paths:
        logger.warn(f"Could not load file data from {path}")
    return pd.concat(dfs, ignore_index=True)  # type: ignore
