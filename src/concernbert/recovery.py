# This is so ugly because it was directly converted from a Jupyter Notebook.
import logging
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from concernbert.embeddings import load_caching_embedder
from concernbert.selection import (
    EntityEdgeSet,
    EntityTree,
    open_db,
)
from concernbert.semantic import MyBert, MyCorpus, MyDoc2Vec, MyLda, MyLsi


def to_combinations(items: list[str], group_size: int) -> list[list[str]]:
    num_groups = len(items) // group_size
    return [items[i * group_size : (i + 1) * group_size] for i in range(num_groups)]


def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """Normalize each vector in X to have a unit length, handle zero-norm vectors."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    # Prevent division by zero by replacing zero norms with 1 (to avoid NaNs)
    norms[norms == 0] = 1

    return X / norms


MODEL_V0 = "_models/CodeBERTa-small-v1"
MODEL_V3 = "_models/EntityBERT-v3_train_nonldl-lr5e5-2_83-e3"
MODEL_V5 = "_models/EntityBERT-v5_train_nonldl-lr5e5-2_80-e3-C2"
CACHE_DIR = "_cache"
OUTPUT_CSV = "_data/recovery5.csv"
OUTPUT_STATE = "_data/recovery_state5.pkl"
MAX_PROJECTS = 5
MAX_FILES_PER_PROJECT = 100
BATCH_SIZE = 16
NUM_KMEANS_RUNS = 4
GROUP_SIZES = [2, 3]
FILES_DF = pd.read_csv("_data/files_test_part.csv")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

CLUSTERING_SEEDS = list(np.random.randint(2**32, size=NUM_KMEANS_RUNS))


def run_split_test(emb_dict, group: list[str], *, normalize: bool) -> float:
    embs: list[np.ndarray] = []
    true: list[int] = []
    for i, filename in enumerate(group):
        emb = emb_dict[filename]
        embs.extend(emb)
        true.extend([i] * emb.shape[0])
    mat = np.vstack(embs)
    if normalize:
        mat = normalize_vectors(mat)
    try:
        nmis: list[float] = []
        for seed in CLUSTERING_SEEDS:
            kmeans = KMeans(n_clusters=len(group), random_state=seed)
            pred: list[int] = list(kmeans.fit(mat).labels_)
            nmi = normalized_mutual_info_score(pred, true)
            nmis.append(float(nmi))
        return float(np.mean(nmis))
    except Exception as e:
        print(f"WARN: Exception encountered during eval: {e}")
        return 0.0


def run_split_test_spectral(emb_dict, group: list[str]) -> float:
    embs: list[np.ndarray] = []
    true: list[int] = []
    for i, filename in enumerate(group):
        emb = emb_dict[filename]
        embs.extend(emb)
        true.extend([i] * emb.shape[0])
    mat = np.vstack(embs)
    sim_matrix = cosine_similarity(mat)
    try:
        nmis: list[float] = []
        for seed in CLUSTERING_SEEDS:
            clustering = SpectralClustering(
                n_clusters=len(group),
                affinity='precomputed',
                random_state=seed
            )
            pred: list[int] = list(clustering.fit_predict(sim_matrix))
            nmi = normalized_mutual_info_score(pred, true)
            nmis.append(float(nmi))
        return float(np.mean(nmis))
    except Exception as e:
        print(f"WARN: Exception encountered during spectral eval: {e}")
        return 0.0


random.seed(SEED)
db_paths = list(sorted(FILES_DF["db_path"].unique()))
random.shuffle(db_paths)
db_paths = db_paths[:MAX_PROJECTS]
print(db_paths)

FILES_DF = FILES_DF[FILES_DF["db_path"].isin(db_paths)]
FILES_DF = FILES_DF[~FILES_DF["is_ldl"]]

print("Training models...")
embedder_v0 = load_caching_embedder(MODEL_V0, CACHE_DIR, BATCH_SIZE)
embedder_v3 = load_caching_embedder(MODEL_V3, CACHE_DIR, BATCH_SIZE)
embedder_v5 = load_caching_embedder(MODEL_V5, CACHE_DIR, BATCH_SIZE)

lsis: dict[str, dict[int, MyLsi]] = defaultdict(dict)
ldas: dict[str, dict[int, MyLda]] = defaultdict(dict)
d2vs: dict[str, dict[int, MyDoc2Vec]] = defaultdict(dict)
bert_v0s: dict[str, MyBert] = dict()
bert_v3s: dict[str, MyBert] = dict()
bert_v5s: dict[str, MyBert] = dict()
filenames: dict[str, list[str]] = defaultdict(list)

bar = tqdm(total=len(FILES_DF))
for db_path, group_df in FILES_DF.groupby("db_path"):
    db_path = str(db_path)
    with open_db(str(db_path)) as conn:
        trees = EntityTree.load_from_db(conn.cursor())
        edge_set = EntityEdgeSet.load_from_db(conn.cursor())
        logging.info(f"Collecting corpus for {db_path}...")
        files_iter = ((t.filename(), t.text()) for t in trees.values())
        corpus = MyCorpus(str(db_path), CACHE_DIR, files_iter, preceding_comments=True)
        dims: list[int] = [10, 64, 256, 768]
        for dim in dims:
            logging.info(f"Running LSI-{dim}...")
            lsis[db_path][dim] = MyLsi(corpus, dim=dim, cache_dir=CACHE_DIR)
        for dim in dims:
            logging.info(f"Running LDA-{dim}...")
            ldas[db_path][dim] = MyLda(corpus, dim=dim, cache_dir=CACHE_DIR)
        for dim in dims:
            logging.info(f"Running D2V-{dim}...")
            d2vs[db_path][dim] = MyDoc2Vec(corpus, dim=dim, cache_dir=CACHE_DIR)
        bert_v0s[db_path] = MyBert(corpus, embedder_v0)
        bert_v3s[db_path] = MyBert(corpus, embedder_v3)
        bert_v5s[db_path] = MyBert(corpus, embedder_v5)
        for _, input_row in group_df.iterrows():
            bar.update()
            tree = trees[input_row["filename"]]  # type: ignore
            cls = tree.standard_class()
            if cls is None:
                continue
            members = tree.children(cls.id)
            m1 = {m.id for m in members if m.kind == "Method"}
            m2 = {m.id for m in members if m.kind == "Constructor"}
            if len(m1 | m2) < 2:
                continue
            filenames[db_path].append(tree.filename())


print("Truncating number of files per project...")
for db_path, names in filenames.items():
    print(f"{db_path}: {len(names)} files")
    random.shuffle(names)
    del names[MAX_FILES_PER_PROJECT:]


print("Embedding...")
lsi_embeddings: dict[str, dict[int, dict[str, np.ndarray]]] = dict()
lda_embeddings: dict[str, dict[int, dict[str, np.ndarray]]] = dict()
d2v_embeddings: dict[str, dict[int, dict[str, np.ndarray]]] = dict()
bert_v0_embeddings: dict[str, dict[str, np.ndarray]] = dict()
bert_v3_embeddings: dict[str, dict[str, np.ndarray]] = dict()
bert_v5_embeddings: dict[str, dict[str, np.ndarray]] = dict()

for db_path, names in filenames.items():
    print()
    print(db_path)
    lsi_embeddings[db_path] = dict()
    lda_embeddings[db_path] = dict()
    d2v_embeddings[db_path] = dict()
    bert_v0_embeddings[db_path] = dict()
    bert_v3_embeddings[db_path] = dict()
    bert_v5_embeddings[db_path] = dict()
    print("LSI")
    for dim, lsi in lsis[db_path].items():
        lsi_embeddings[db_path][dim] = dict()
        for name in names:
            lsi_embeddings[db_path][dim][name] = lsi.embed(name)
    print("LDA")
    for dim, lda in ldas[db_path].items():
        lda_embeddings[db_path][dim] = dict()
        for name in names:
            lda_embeddings[db_path][dim][name] = lda.embed(name)
    print("Doc2Vec")
    for dim, d2v in d2vs[db_path].items():
        d2v_embeddings[db_path][dim] = dict()
        for name in names:
            d2v_embeddings[db_path][dim][name] = d2v.embed(name)
    print("BERT v0")
    print(len(names))
    for name in names:
        bert_v0_embeddings[db_path][name] = bert_v0s[db_path].embed(name)
    print("BERT v3")
    print(len(names))
    for name in names:
        bert_v3_embeddings[db_path][name] = bert_v3s[db_path].embed(name)
    print("BERT v5")
    print(len(names))
    for name in names:
        bert_v5_embeddings[db_path][name] = bert_v5s[db_path].embed(name)


print("Building tests...")
tests: dict[str, dict[int, list[list[str]]]] = dict()
n_tests = 0

for db_path, names in tqdm(filenames.items()):
    tests[db_path] = dict()
    for group_size in GROUP_SIZES:
        shuffled_names = list(names)
        random.shuffle(shuffled_names)
        combinations = to_combinations(shuffled_names, group_size)
        tests[db_path][group_size] = combinations
        n_tests += len(combinations)

print(f"# of tests: {n_tests}")

print("Running tests...")
results = []

for i, (db_path, db_tests) in enumerate(tests.items()):
    print(f"[{i + 1}/{len(tests)}] Running on {db_path}...")
    for group_size, groups in db_tests.items():
        for group_i, group in enumerate(groups):
            # LSI
            for dim, emb_dict in lsi_embeddings[db_path].items():
                nmi = run_split_test(emb_dict, group, normalize=False)
                results.append([db_path, group_size, group_i, "LSI", str(dim), "kmeans", nmi])
                nmi = run_split_test(emb_dict, group, normalize=True)
                results.append([db_path, group_size, group_i, "LSI", str(dim), "kmeans-sphere", nmi])
                nmi = run_split_test_spectral(emb_dict, group)
                results.append([db_path, group_size, group_i, "LSI", str(dim), "spectral", nmi])
            
            # LDA
            for dim, emb_dict in lda_embeddings[db_path].items():
                nmi = run_split_test(emb_dict, group, normalize=False)
                results.append([db_path, group_size, group_i, "LDA", str(dim), "kmeans", nmi])
                nmi = run_split_test(emb_dict, group, normalize=True)
                results.append([db_path, group_size, group_i, "LDA", str(dim), "kmeans-sphere", nmi])
                nmi = run_split_test_spectral(emb_dict, group)
                results.append([db_path, group_size, group_i, "LDA", str(dim), "spectral", nmi])
            
            # Doc2Vec
            for dim, emb_dict in d2v_embeddings[db_path].items():
                nmi = run_split_test(emb_dict, group, normalize=False)
                results.append([db_path, group_size, group_i, "D2V", str(dim), "kmeans", nmi])
                nmi = run_split_test(emb_dict, group, normalize=True)
                results.append([db_path, group_size, group_i, "D2V", str(dim), "kmeans-sphere", nmi])
                nmi = run_split_test_spectral(emb_dict, group)
                results.append([db_path, group_size, group_i, "D2V", str(dim), "spectral", nmi])
            
            # BERT v0
            emb_v0_dict = bert_v0_embeddings[db_path]
            nmi = run_split_test(emb_v0_dict, group, normalize=False)
            results.append([db_path, group_size, group_i, "BERTv0", "768", "kmeans", nmi])
            nmi = run_split_test(emb_v0_dict, group, normalize=True)
            results.append([db_path, group_size, group_i, "BERTv0", "768", "kmeans-sphere", nmi])
            nmi = run_split_test_spectral(emb_v0_dict, group)
            results.append([db_path, group_size, group_i, "BERTv0", "768", "spectral", nmi])

            # BERT v3
            emb_v3_dict = bert_v3_embeddings[db_path]
            nmi = run_split_test(emb_v3_dict, group, normalize=False)
            results.append([db_path, group_size, group_i, "BERTv3", "768", "kmeans", nmi])
            nmi = run_split_test(emb_v3_dict, group, normalize=True)
            results.append([db_path, group_size, group_i, "BERTv3", "768", "kmeans-sphere", nmi])
            nmi = run_split_test_spectral(emb_v3_dict, group)
            results.append([db_path, group_size, group_i, "BERTv3", "768", "spectral", nmi])

            # BERT v5
            emb_v5_dict = bert_v5_embeddings[db_path]
            nmi = run_split_test(emb_v5_dict, group, normalize=False)
            results.append([db_path, group_size, group_i, "BERTv5", "768", "kmeans", nmi])
            nmi = run_split_test(emb_v5_dict, group, normalize=True)
            results.append([db_path, group_size, group_i, "BERTv5", "768", "kmeans-sphere", nmi])
            nmi = run_split_test_spectral(emb_v5_dict, group)
            results.append([db_path, group_size, group_i, "BERTv5", "768", "spectral", nmi])



print("Collecting results...")
df = pd.DataFrame(results, columns=["db_path", "group_size", "group_i", "model", "dim", "alg", "nmi"])
print("Writing CSV...")
df.to_csv(OUTPUT_CSV)
print("Writing state...")
state = {
    "tests": tests,
    "filenames": filenames,
    "lsi_embeddings": lsi_embeddings,
    "lda_embeddings": lda_embeddings,
    "d2v_embeddings": d2v_embeddings,
    "bert_v0_embeddings": bert_v0_embeddings,
    "bert_v3_embeddings": bert_v3_embeddings,
    "bert_v5_embeddings": bert_v5_embeddings,
}
with open(OUTPUT_STATE, "wb") as f:
    pickle.dump(state, f)
