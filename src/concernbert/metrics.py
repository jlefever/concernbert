import itertools as it
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy as sp
from ordered_set import OrderedSet as oset
from scipy.spatial.distance import cdist, euclidean
from tqdm import tqdm

from concernbert.embeddings import Embedder, load_caching_embedder
from concernbert.selection import (
    EntityEdgeSet,
    EntityGraph,
    EntityNodeSet,
    EntityTree,
    calc_canonical,
    open_db,
)
from concernbert.semantic import MyBert, MyCorpus, MyDoc2Vec, MyLsi


def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """Normalize each vector in X to have a unit length, handle zero-norm vectors."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Prevent division by zero by replacing zero norms with 1 (to avoid NaNs)
    norms[norms == 0] = 1
    
    return X / norms


def to_centroid(X: np.ndarray) -> np.ndarray:
    """Calculate the centroid (mean vector) of X."""
    return np.mean(X, axis=0)


# From https://stackoverflow.com/a/30305181
def to_geometric_median(X: np.ndarray, eps=1e-8) -> np.ndarray:
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def to_medoid(X: np.ndarray) -> np.ndarray:
    return X[np.argmin(np.sum(cdist(X, X), axis=0))]


def to_marginal_median(X: np.ndarray) -> np.ndarray:
    return np.median(X, axis=0)


def to_aad(X: np.ndarray) -> float:
    centroid = to_centroid(X)
    centroid_residuals = cdist(X, [centroid])
    return float(np.mean(centroid_residuals))


def to_unit_aad(X: np.ndarray) -> float:
    X_normalized = normalize_vectors(X)
    centroid = to_centroid(X_normalized)
    centroid_residuals = cdist(X_normalized, [centroid])
    return float(np.mean(centroid_residuals))


def to_sim_mat(embeddings: np.ndarray, *, euclidean: bool = False) -> np.ndarray:
    if euclidean:
        dists = cdist(embeddings, embeddings, metric="euclidean")
        return 1 / (1 + dists)
    dists = cdist(embeddings, embeddings, metric="cosine")
    return (dists - 1) * -1


def to_dist_mat(embeddings: np.ndarray) -> np.ndarray:
    return cdist(embeddings, embeddings, metric="euclidean")


def to_acsm(sim_mat: np.ndarray) -> float:
    "Returns ACSM (defined by Marcus and Poshyvanyk)"
    return float(np.mean(sim_mat[np.triu_indices(len(sim_mat), k=1)]))


def to_acosm(sim_mat: np.ndarray) -> float:
    "Returns ACOSM (defined by Miholca)"
    return to_acsm(sim_mat)


def to_c3(acsm: float) -> float:
    "Returns C3 (defined by Marcus and Poshyvanyk)"
    return max(acsm, 0)


def to_cocc(acosm: float) -> float:
    "Returns COCC (defined by Miholca)"
    return to_c3(acosm)


def to_lcsm(sim_mat: np.ndarray, acsm: float) -> int:
    "Returns LCSM (defined by Marcus and Poshyvanyk)"
    n = sim_mat.shape[0]
    neighbors: list[set[int]] = []
    for i in range(n):
        indices = np.argwhere(sim_mat[i] > acsm).flatten()
        neighbors.append(set(j for j in indices if j != i))
    if all(len(n) == 0 for n in neighbors):
        return 0
    p, q = 0, 0
    for a, b in it.combinations(neighbors, 2):
        if len(a & b) == 0:
            p += 1
        else:
            q += 1
    return max(0, p - q)


def to_lcosm(sim_mat: np.ndarray, acosm: float) -> float | None:
    "Returns LCOSM (defined by Miholca)"
    n = sim_mat.shape[0]
    if n < 2:
        return None
    lcsm = to_lcsm(sim_mat, acosm)
    return lcsm / ((n * (n - 1)) / 2)


@dataclass
class ModelBasedCohesion:
    mean_dist_to_centroid: float
    median_dist_to_centroid: float
    std_dist_to_centroid: float
    mean_dist_to_geometric_median: float
    median_dist_to_geometric_median: float
    std_dist_to_geometric_median: float
    mean_dist_to_medoid: float
    median_dist_to_medoid: float
    std_dist_to_medoid: float
    mean_dist_to_marginal_median: float
    median_dist_to_marginal_median: float
    std_dist_to_marginal_median: float


def to_model_based_cohesion(embeddings: np.ndarray) -> ModelBasedCohesion:
    centroid = to_centroid(embeddings)
    geometric_median = to_geometric_median(embeddings)
    medoid = to_medoid(embeddings)
    marginal_median = to_marginal_median(embeddings)

    centroid_residuals = cdist(embeddings, [centroid])
    geometric_median_residuals = cdist(embeddings, [geometric_median])
    medoid_residuals = cdist(embeddings, [medoid])
    marginal_median_residuals = cdist(embeddings, [marginal_median])

    return ModelBasedCohesion(
        mean_dist_to_centroid=float(np.mean(centroid_residuals)),
        median_dist_to_centroid=float(np.median(centroid_residuals)),
        std_dist_to_centroid=float(np.std(centroid_residuals)),
        mean_dist_to_geometric_median=float(np.mean(geometric_median_residuals)),
        median_dist_to_geometric_median=float(np.median(geometric_median_residuals)),
        std_dist_to_geometric_median=float(np.std(geometric_median_residuals)),
        mean_dist_to_medoid=float(np.mean(medoid_residuals)),
        median_dist_to_medoid=float(np.median(medoid_residuals)),
        std_dist_to_medoid=float(np.std(medoid_residuals)),
        mean_dist_to_marginal_median=float(np.mean(marginal_median_residuals)),
        median_dist_to_marginal_median=float(np.median(marginal_median_residuals)),
        std_dist_to_marginal_median=float(np.std(marginal_median_residuals)),
    )


def calc_metrics_row(
    tree: EntityTree,
    subgraph: EntityGraph,
    embedder: Embedder,
    lsis: dict[str, MyLsi],
    d2vs: dict[str, MyDoc2Vec],
    bert: MyBert,
) -> dict[str, Any]:
    row: dict[str, Any] = dict()

    # "Members" are the methods and attributes of the standard class
    row["Members"] = len(subgraph.nodes)
    row["Methods"] = len(subgraph.nodes.methods())
    row["Fields"] = len(subgraph.nodes.attributes())

    # Our metric (calculated with methods + fields)
    texts = [tree.entity_text(m.id) for m in subgraph.nodes]
    embeddings_dict = embedder.embed(texts, pbar=False)
    embeddings = np.array([embeddings_dict[t] for t in texts])
    row["CDI"] = to_aad(embeddings)

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

    # If there are no less than two methods, the remaining metrics are undefined.
    if len(subgraph.nodes.methods()) < 2:
        return row

    # LSI embeddings and similarity matrices (used by Marcus and Poshyvanyk)
    lsi_embeddings: dict[str, np.ndarray] = {}
    for name, lsi in lsis.items():
        lsi_embeddings[name] = lsi.embed(tree.filename())
    lsi_sim_mats: dict[str, np.ndarray] = {}
    for name, emb in lsi_embeddings.items():
        lsi_sim_mats[name] = to_sim_mat(emb)

    # Doc2Vec embeddings and similarity matrices (used by Miholca)
    d2v_embeddings: dict[str, np.ndarray] = {}
    for name, d2v in d2vs.items():
        d2v_embeddings[name] = d2v.embed(tree.filename())
    d2v_sim_mats: dict[str, np.ndarray] = {}
    for name, emb in d2v_embeddings.items():
        d2v_sim_mats[name] = to_sim_mat(emb)

    # BERT embeddings and similarity matrices (used by us)
    # These are used to make a comparison with prior work that only uses methods
    bert_embeddings = bert.embed(tree.filename())
    bert_sim_mat = to_sim_mat(bert_embeddings)

    # AAD
    for name, emb in lsi_embeddings.items():
        # We use unit_aad instead because LSI space typically doesn't use magnitude (usually cosine similarity)
        row[f"AAD(LSI-{name})"] = to_unit_aad(emb)
    for name, emb in d2v_embeddings.items():
        # Same for Doc2Vec
        row[f"AAD(D2V-{name})"] = to_unit_aad(emb)
    row["AAD(BERT)"] = to_aad(bert_embeddings)

    # Negative C3
    for name, sim_mat in lsi_sim_mats.items():
        row[f"NC3(LSI-{name})"] = -1 * to_c3(to_acsm(sim_mat))
    for name, sim_mat in d2v_sim_mats.items():
        row[f"NC3(D2V-{name})"] = -1 * to_c3(to_acsm(sim_mat))
    row["NC3(BERT)"] = -1 * to_c3(to_acsm(bert_sim_mat))

    # LCSM
    for name, sim_mat in lsi_sim_mats.items():
        row[f"LCSM(LSI-{name})"] = to_lcsm(sim_mat, to_acsm(sim_mat))
    for name, sim_mat in d2v_sim_mats.items():
        row[f"LCSM(D2V-{name})"] = to_lcsm(sim_mat, to_acsm(sim_mat))
    row["LCSM(BERT)"] = to_lcsm(bert_sim_mat, to_acsm(bert_sim_mat))
    
    # LCOSM
    for name, sim_mat in lsi_sim_mats.items():
        row[f"LCOSM(LSI-{name})"] = to_lcosm(sim_mat, to_acsm(sim_mat))
    for name, sim_mat in d2v_sim_mats.items():
        row[f"LCOSM(D2V-{name})"] = to_lcosm(sim_mat, to_acsm(sim_mat))
    row["LCOSM(BERT)"] = to_lcosm(bert_sim_mat, to_acsm(bert_sim_mat))

    return row


def calc_metrics_df(
    files_df: pd.DataFrame, model: str, cache_dir: str, batch_size: int, *, pbar: bool
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Load embedder
    embedder = load_caching_embedder(model, cache_dir, batch_size)

    bar = tqdm(total=len(files_df), disable=not pbar)
    for db_path, group_df in files_df.groupby("db_path"):
        with open_db(str(db_path)) as conn:
            trees = EntityTree.load_from_db(conn.cursor())
            edge_set = EntityEdgeSet.load_from_db(conn.cursor())
            logging.info(f"Collecting corpus for {db_path} ({len(trees)} files)...")
            files_iter = ((t.filename(), t.text()) for t in trees.values())

            # Throw out any very large files
            files_iter = ((f, t) for f, t in files_iter if len(t) <= 2_000_000)
            corpus_c = MyCorpus(str(db_path), cache_dir, files_iter, preceding_comments=True)
            corpus_nc = MyCorpus(str(db_path), cache_dir, files_iter, preceding_comments=False)
            dims: list[int] = [10, 64, 256, 768]
            lsis: dict[str, MyLsi] = {}
            for dim, c in it.product(dims, ["C", "NC"]):
                logging.info(f"Running LSI-{dim}-{c}...")
                if c == "C":
                    lsis[f"{dim}-{c}"] = MyLsi(corpus_c, dim=dim, cache_dir=cache_dir)
                elif c == "NC":
                    lsis[f"{dim}-{c}"] = MyLsi(corpus_nc, dim=dim, cache_dir=cache_dir)
            d2vs: dict[str, MyDoc2Vec] = {}
            for dim, c in it.product(dims, ["C", "NC"]):
                logging.info(f"Running D2V-{dim}-{c}...")
                if c == "C":
                    d2vs[f"{dim}-{c}"] = MyDoc2Vec(corpus_c, dim=dim, cache_dir=cache_dir)
                elif c == "NC":
                    d2vs[f"{dim}-{c}"] = MyDoc2Vec(corpus_nc, dim=dim, cache_dir=cache_dir)
            bert = MyBert(corpus_nc, embedder)
            for _, input_row in group_df.iterrows():
                bar.update()
                tree = trees[input_row["filename"]]  # type: ignore
                cls = tree.standard_class()
                if cls is None:
                    continue
                members = oset(tree.children(cls.id))
                member_ids = set(m.id for m in members)
                subgraph = EntityGraph(
                    EntityNodeSet(members), edge_set.subset(member_ids)
                )
                try:
                    row = input_row.to_dict()
                    metrics_row = calc_metrics_row(
                        tree, subgraph, embedder, lsis, d2vs, bert
                    )
                    row.update(metrics_row)
                    rows.append(row)
                except Exception as e:
                    logging.warning(f"Skipping {tree.filename()} due to exception ({e})")
    return pd.DataFrame.from_records(rows)
