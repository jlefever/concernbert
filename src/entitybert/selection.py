import itertools as it
import math
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from ordered_set import OrderedSet as oset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, floyd_warshall
from tqdm import tqdm

_SELECT_FILES = """
    SELECT
        F.filename,
        CO.loc,
        CO.lloc,
        COUNT(DISTINCT F.entity_id) AS entities,
        COUNT(DISTINCT CH.commit_id) AS commits
    FROM filenames F
    LEFT JOIN changes CH ON CH.simple_id = F.simple_id
    LEFT JOIN contents CO ON CO.content_id = F.content_id
    GROUP BY F.filename
    ORDER BY F.filename
"""


_SELECT_ENTITIES = """
    SELECT
        F.filename,
        HEX(E.id) AS id,
        NULLIF(HEX(parent_id), '') AS parent_id,
        E.name,
        E.kind,
        E.start_byte,
        E.start_row,
        E.start_column,
        E.end_byte,
        E.end_row,
        E.end_column
    FROM entities E
    JOIN filenames F ON F.entity_id = E.id
    ORDER BY F.filename, E.parent_id, E.id
"""


@dataclass(unsafe_hash=True)
class EntityDto:
    filename: str
    id: str
    parent_id: str | None
    name: str
    kind: str
    start_byte: int
    start_row: int
    start_column: int
    end_byte: int
    end_row: int
    end_column: int
    _modifiers: frozenset[str] | None = None  # Hacky...

    def text(self, content: bytes) -> str:
        return content[self.start_byte : self.end_byte].decode()

    def is_file(self) -> bool:
        return self.kind == "File"

    def is_class(self) -> bool:
        return self.kind == "Class"

    def is_attribute(self) -> bool:
        return self.kind == "Field"

    def is_constructor(self) -> bool:
        return self.kind == "Constructor"

    def is_method(self) -> bool:
        return self.kind == "Method" or self.is_constructor()

    def is_abstract(self) -> bool:
        if self._modifiers is None:
            raise ValueError()
        return "abstract" in self._modifiers

    def is_public(self) -> bool:
        if self._modifiers is None:
            raise ValueError()
        return "public" in self._modifiers

    def is_static(self) -> bool:
        if self._modifiers is None:
            raise ValueError()
        return "static" in self._modifiers


def _select_entities(cursor: Cursor) -> Iterable[EntityDto]:
    cursor.execute(_SELECT_ENTITIES)
    yield from (EntityDto(**r) for r in cursor.fetchall())


_SELECT_DEPS = """
    SELECT DISTINCT
        HEX(D.src) AS src,
        HEX(D.tgt) AS tgt
    FROM deps D
    WHERE D.src <> D.tgt
    ORDER BY D.src, D.tgt
"""


@dataclass
class _DepDto:
    src: str
    tgt: str


def _select_deps(cursor: Cursor) -> Iterable[_DepDto]:
    cursor.execute(_SELECT_DEPS)
    yield from (_DepDto(**r) for r in cursor.fetchall())


_SELECT_CONTENTS = """
    SELECT E.name AS filename, C.content
    FROM entities E
    JOIN contents C ON C.content_id = E.content_id
    WHERE E.kind = 'File'
    ORDER BY E.name
"""


@dataclass
class _ContentDto:
    filename: str
    content: bytes


def _select_contents(cursor: Cursor) -> Iterable[_ContentDto]:
    cursor.execute(_SELECT_CONTENTS)
    for row in cursor.fetchall():
        row["content"] = row["content"].encode()
        yield _ContentDto(**row)


_INVALID_ENDINGS = [
    "-info.java",
    "Benchmark.java",
    "Benchmarks.java",
    "Demo.java",
    "Demos.java",
    "Example.java",
    "Examples.java",
    "Exercise.java",
    "Exercises.java",
    "Guide.java",
    "Guides.java",
    "Sample.java",
    "Samples.java",
    "Scenario.java",
    "Scenarios.java",
    "Test.java",
    "Tests.java",
    "Tutorial.java",
    "Tutorials.java",
]


_INVALID_SEGMENTS = set(
    [
        "benchmark",
        "benchmarks",
        "demo",
        "demos",
        "example",
        "examples",
        "exercise",
        "exercises",
        "gen",
        "generated",
        "guide",
        "guides",
        "integration-test",
        "integration-tests",
        "quickstart",
        "quickstarts",
        "sample",
        "samples",
        "scenario",
        "scenarios",
        "test",
        "testkit",
        "tests",
        "tutorial",
        "tutorials",
    ]
)


def _is_filename_valid(filename: str) -> bool:
    if any(filename.endswith(e) for e in _INVALID_ENDINGS):
        return False
    segments = filename.lower().split("/")
    return not any(s in _INVALID_SEGMENTS for s in segments)


def _load_files_df(conn: Connection) -> pd.DataFrame:
    files_df = pd.read_sql(_SELECT_FILES, conn, index_col="filename")
    files_df = files_df[[_is_filename_valid(str(f)) for f in files_df.index]]
    if files_df.isnull().values.any():
        raise RuntimeError("DataFrame contains NaN values.")
    files_df.sort_index(inplace=True)
    return files_df


def list_db_paths(dbs_file: str) -> list[str]:
    return Path(dbs_file).read_text().splitlines()


def load_multi_files_df(db_paths: list[str]) -> pd.DataFrame:
    dfs = []
    for db_path in tqdm(db_paths):
        with sqlite3.connect(db_path) as conn:
            df = _load_files_df(conn)
            df.reset_index(inplace=True)
            df.insert(0, "db_path", db_path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def insert_ldl_cols(files_df: pd.DataFrame, *, q: float = 0.8):
    is_large1 = files_df["lloc"] >= files_df["lloc"].quantile(q)
    is_large2 = files_df["entities"] >= files_df["entities"].quantile(q)
    files_df["is_large"] = is_large1 | is_large2
    thresholds = dict(files_df.groupby("db_path")["commits"].quantile(q))
    files_df["is_change_prone"] = files_df["commits"] >= [
        thresholds[p] for p in files_df["db_path"]
    ]
    files_df["is_ldl"] = files_df["is_large"] & files_df["is_change_prone"]


class EntityTree:
    def __init__(self, filename: str, content: bytes):
        self._filename = filename
        self._content = content
        self._entities: dict[str, EntityDto] = dict()
        self._children: defaultdict[str | None, list[str]] = defaultdict(list)

    @staticmethod
    def load_from_db(cursor: Cursor) -> dict[str, "EntityTree"]:
        contents: dict[str, bytes] = dict()
        trees: dict[str, EntityTree] = dict()
        # Collect contents
        groups = it.groupby(_select_contents(cursor), key=lambda x: x.filename)
        for filename, group in groups:
            group = list(group)
            if len(group) != 1:
                msg = f"found {len(group)} contents. Expected 1. ({filename})"
                raise RuntimeError(msg)
            contents[filename] = group[0].content
        # Collect entities
        groups = it.groupby(_select_entities(cursor), key=lambda x: x.filename)
        for filename, group in groups:
            tree = EntityTree(filename, contents[filename])
            for entity in group:
                tree._add_entity(entity)
            tree._determine_modifiers()
            trees[filename] = tree
        return trees

    def _add_entity(self, entity: EntityDto):
        if entity.parent_id is None and entity.kind != "File":
            raise ValueError(f"Entity (id: {entity.id}) is a root but not a File")
        self._entities[entity.id] = entity
        self._children[entity.parent_id].append(entity.id)

    def _determine_modifiers(self) -> None:
        for entity in self._entities.values():
            # Five is the maximum number of keywords that could appear before an
            # identifier in a Java method.
            keywords = entity.text(self._content).split()
            entity._modifiers = frozenset(k.lower() for k in keywords[0:5])

    def __getitem__(self, id: str) -> EntityDto:
        return self._entities[id]

    def filename(self) -> str:
        return self._filename

    def is_leaf(self, id: str) -> bool:
        "Returns true if this entity has no children"
        return len(self._children[id]) == 0

    def children(self, id: str | None) -> list[EntityDto]:
        return [self._entities[c] for c in self._children[id]]

    def leaf_children(self, id: str) -> list[str]:
        "Returns a list of leaf children for this entity"
        return [c for c in self._children[id] if self.is_leaf(c)]

    def leaf_siblings(self) -> list[list[str]]:
        return [self.leaf_children(id) for id in self._entities]

    def nontrivial_leaf_siblings(self) -> list[list[str]]:
        return [s for s in self.leaf_siblings() if len(s) > 1]

    def standard_class(self) -> EntityDto | None:
        """
        Returns the id of the standard class if one exists in this file.

        A standard class occurs when a file has exactly one root entity. This
        root entity is a class with at least two children. All children must be
        either attributes or methods.
        """
        roots = self.children(None)
        if len(roots) != 1:
            # Root is more than one element (should not be possible)
            return None
        file = roots[0]
        if not file.is_file():
            # Root is not a file (should not be possible)
            return None
        file_children = self.children(file.id)
        if len(file_children) != 1:
            # More than one element directly below file (should not be possible (in Java))
            return None
        cls = file_children[0]
        if not cls.is_class():
            # Top-level entity is not a class
            return None
        cls_children = self.children(cls.id)
        if len(cls_children) < 2:
            # Top-level class has less than two children
            return None
        if any(not (c.is_attribute() or c.is_method()) for c in cls_children):
            # Top-level class has direct children that are not attributes or methods
            return None
        return cls

    def text(self) -> str:
        return self._content.decode()

    def entity_text(self, id: str) -> str:
        return self._entities[id].text(self._content)

    def to_entity_row(self, db_path: str | None, entity_id: str) -> dict[str, Any]:
        entity = self._entities[entity_id]
        if entity.parent_id is None:
            raise RuntimeError("root entities cannot be made into rows")
        parent = self._entities[entity.parent_id]
        row: dict[str, Any] = dict()
        row["db_path"] = db_path
        row["filename"] = self._filename
        row["parent_id"] = parent.id
        row["parent_name"] = parent.name
        row["parent_kind"] = parent.kind
        row["id"] = entity.id
        row["name"] = entity.name
        row["kind"] = entity.kind
        row["start_byte"] = entity.start_byte
        row["start_row"] = entity.start_row
        row["start_column"] = entity.start_column
        row["end_byte"] = entity.end_byte
        row["end_row"] = entity.end_row
        row["end_column"] = entity.end_column
        row["content"] = self.entity_text(entity.id)
        return row

    def to_entities_df(self, db_path: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for siblings in self.nontrivial_leaf_siblings():
            for id in siblings:
                rows.append(self.to_entity_row(db_path, id))
        return pd.DataFrame.from_records(rows)  # type: ignore


def iter_entity_trees(
    files_df: pd.DataFrame, *, pbar: bool
) -> Iterator[tuple[pd.Series, EntityTree]]:
    bar = tqdm(total=len(files_df), disable=not pbar)
    for db_path, group_df in files_df.groupby("db_path"):
        with open_db(str(db_path)) as conn:
            trees = EntityTree.load_from_db(conn.cursor())
            for _, row in group_df.iterrows():
                bar.update()
                tree = trees[row["filename"]]  # type: ignore
                yield (row, tree)


def extract_entities_df(files_df: pd.DataFrame, *, pbar: bool) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for row, tree in iter_entity_trees(files_df, pbar=pbar):
        dfs.append(tree.to_entities_df(row["db_path"]))
    return pd.concat(dfs, ignore_index=True)


def count_components(mat: np.ndarray) -> int:
    n_components, _ = connected_components(mat, directed=False)
    return n_components


def to_sym_adj_mat(mat: np.ndarray) -> np.ndarray:
    "Returns a symmetric binary matrix given an possibly asymmetric binary matrix."
    return np.logical_or(mat, mat.T).astype(int)


def to_trans_closure(mat: np.ndarray) -> np.ndarray:
    "Returns the transitive closure of a binary adjacency matrix. Will include self-loops."
    reachability_mat = floyd_warshall(mat, directed=True, unweighted=True)
    return (reachability_mat != np.inf).astype(int)


class EntityEdgeSet:
    def __init__(self, pairs: set[tuple[str, str]]):
        self._pairs = set((a, b) for a, b in pairs if a != b)
        self._incoming: defaultdict[str, set[str]] = defaultdict(set)
        self._outgoing: defaultdict[str, set[str]] = defaultdict(set)
        for src, tgt in self._pairs:
            self._incoming[tgt].add(src)
            self._outgoing[src].add(tgt)

    @staticmethod
    def load_from_db(cursor: Cursor) -> "EntityEdgeSet":
        return EntityEdgeSet({(d.src, d.tgt) for d in _select_deps(cursor)})

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._pairs)

    @property
    def pairs(self) -> set[tuple[str, str]]:
        return self._pairs

    def outgoing(self, node: str) -> set[str]:
        return self._outgoing[node]

    def incoming(self, node: str) -> set[str]:
        return self._incoming[node]

    def adjacent(self, node: str) -> set[str]:
        return self._incoming[node] | self._outgoing[node]

    def subset(self, nodes: set[str]) -> "EntityEdgeSet":
        edge_set = set()
        for src in nodes:
            for tgt in self._outgoing[src] & nodes:
                edge_set.add((src, tgt))
        return EntityEdgeSet(edge_set)


class EntityNodeSet:
    def __init__(self, nodes: oset[EntityDto]):
        self._nodes = nodes
        self._nodes_by_id = {n.id: n for n in nodes}
        if len(self._nodes) != len(self._nodes_by_id):
            raise ValueError("duplicate node ids")

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[EntityDto]:
        return iter(self._nodes)

    @property
    def nodes(self) -> oset[EntityDto]:
        return self._nodes

    def get(self, node: int | str | EntityDto) -> EntityDto:
        if isinstance(node, EntityDto):
            return node
        if isinstance(node, int):
            return self._nodes[node]
        return self._nodes_by_id[node]

    def get_ix(self, node: int | str | EntityDto) -> int:
        if isinstance(node, int):
            return node
        if isinstance(node, str):
            node = self._nodes_by_id[node]
        return self._nodes.index(node)

    def get_id(self, node: int | str | EntityDto) -> str:
        if isinstance(node, str):
            return node
        if isinstance(node, int):
            node = self._nodes[node]
        return node.id

    @cache
    def attributes(self) -> set[str]:
        return set(n.id for n in self._nodes if n.is_attribute())

    @cache
    def methods(self) -> set[str]:
        return set(n.id for n in self._nodes if n.is_method())

    @cache
    def abstracts(self) -> set[str]:
        return set(n.id for n in self._nodes if n.is_abstract())

    @cache
    def impl_methods(self) -> set[str]:
        return self.methods() - self.abstracts()

    @cache
    def publics(self) -> set[str]:
        return set(n.id for n in self._nodes if n.is_public())

    @cache
    def constructors(self) -> set[str]:
        return set(n.id for n in self._nodes if n.is_constructor())

    def subset(self, nodes: set[str]) -> "EntityNodeSet":
        return EntityNodeSet(oset(n for n in self.nodes if n in nodes))


class EntityGraph:
    def __init__(self, nodes: EntityNodeSet, edges: EntityEdgeSet):
        self._nodes = nodes
        self._edges = edges

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> EntityNodeSet:
        return self._nodes

    @property
    def edges(self) -> EntityEdgeSet:
        return self._edges

    def get_node(self, node: int | str | EntityDto) -> EntityDto:
        return self._nodes.get(node)

    def get_node_ix(self, node: int | str | EntityDto) -> int:
        return self._nodes.get_ix(node)

    def get_node_id(self, node: int | str | EntityDto) -> str:
        return self._nodes.get_id(node)

    @cache
    def attribute_refs(self, node: int | str | EntityDto) -> set[str]:
        outgoing = self._edges.outgoing(self.get_node_id(node))
        return outgoing & self.nodes.methods()

    def is_edge(self, src: int | str | EntityDto, tgt: int | str | EntityDto) -> bool:
        src_id = self.get_node_id(src)
        tgt_id = self.get_node_id(tgt)
        return tgt_id in self._edges.outgoing(src_id)

    @staticmethod
    def from_adj_mat(nodes: EntityNodeSet, adj_mat: np.ndarray) -> "EntityGraph":
        pairs = set(
            (nodes.get_id(int(i)), nodes.get_id(int(j)))
            for i, j in np.argwhere(adj_mat > 0)
        )
        return EntityGraph(nodes, EntityEdgeSet(pairs))

    def to_adj_mat(self) -> np.ndarray:
        adj_mat = np.zeros((len(self), len(self)))
        for src, tgt in self._edges:
            adj_mat[self.get_node_ix(src), self.get_node_ix(tgt)] = 1
        return adj_mat

    def to_sym_adj_mat(self) -> np.ndarray:
        return to_sym_adj_mat(self.to_adj_mat())

    def to_trans_closure(self) -> "EntityGraph":
        return EntityGraph.from_adj_mat(self.nodes, to_trans_closure(self.to_adj_mat()))

    def subgraph(self, nodes: Iterable[int | str | EntityDto]) -> "EntityGraph":
        entities: oset[EntityDto] = oset(self.get_node(n) for n in nodes)
        entity_ids: set[str] = set(e.id for e in entities)
        return EntityGraph(EntityNodeSet(entities), self._edges.subset(entity_ids))


def calc_graph_a(call_graph: EntityGraph) -> np.ndarray:
    methods = oset(sorted(call_graph.nodes.impl_methods()))
    adj = np.zeros((len(methods), len(methods)))
    for m1, m2 in it.combinations(methods, 2):
        m1_ix = methods.index(m1)
        m2_ix = methods.index(m2)
        a1 = call_graph.attribute_refs(m1)
        a2 = call_graph.attribute_refs(m2)
        if len(a1 & a2) > 0:
            adj[m1_ix, m2_ix] = 1
            adj[m2_ix, m1_ix] = 1
    return adj


def calc_graph_b(call_graph: EntityGraph) -> np.ndarray:
    methods = oset(sorted(call_graph.nodes.impl_methods()))
    adj = np.zeros((len(methods), len(methods)))
    for m1, m2 in it.combinations(methods, 2):
        m1_ix = methods.index(m1)
        m2_ix = methods.index(m2)
        a1 = call_graph.attribute_refs(m1)
        a2 = call_graph.attribute_refs(m2)
        shared_attr = len(a1 & a2) > 0
        forward_edge = call_graph.is_edge(m1, m2)
        backward_edge = call_graph.is_edge(m2, m1)
        if shared_attr or forward_edge or backward_edge:
            adj[m1_ix, m2_ix] = 1
            adj[m2_ix, m1_ix] = 1
    return adj


def calc_graph_c(call_graph: EntityGraph) -> np.ndarray:
    call_graph = call_graph.to_trans_closure()
    methods = call_graph.nodes.impl_methods()
    constructors = call_graph.nodes.constructors()
    publics = call_graph.nodes.publics()
    methods = oset(sorted((methods & publics) - constructors))
    adj = np.zeros((len(methods), len(methods)))
    for m1, m2 in it.combinations(methods, 2):
        m1_ix = methods.index(m1)
        m2_ix = methods.index(m2)
        a1 = call_graph.attribute_refs(m1)
        a2 = call_graph.attribute_refs(m2)
        if len(a1 & a2) > 0:
            adj[m1_ix, m2_ix] = 1
            adj[m2_ix, m1_ix] = 1
    return adj


def calc_lcom1(adj_a: np.ndarray) -> int:
    upper_tri_mask = np.triu(np.ones(adj_a.shape), k=1)
    upper_tri_values = adj_a[upper_tri_mask == 1]
    return np.sum(upper_tri_values == 0)


def calc_lcom2(adj_a: np.ndarray) -> int:
    if np.all(adj_a == 0):
        return 0
    upper_tri_mask = np.triu(np.ones(adj_a.shape), k=1)
    upper_tri_values = adj_a[upper_tri_mask == 1]
    p = np.sum(upper_tri_values == 0)
    q = len(upper_tri_values) - p
    return max(0, p - q)


def calc_lcom3(adj_a: np.ndarray) -> int:
    return count_components(adj_a)


def calc_lcom4(adj_b: np.ndarray) -> int:
    return count_components(adj_b)


def calc_co(adj_b: np.ndarray) -> float | None:
    n_nodes = adj_b.shape[0]
    if n_nodes < 3:
        return None
    upper_tri_mask = np.triu(np.ones(adj_b.shape), k=1)
    upper_tri_values = adj_b[upper_tri_mask == 1]
    n_edges = np.sum(upper_tri_values == 1)
    return (n_edges - (n_nodes - 1)) / ((n_nodes - 1) * (n_nodes - 2))


def calc_tcc(adj_c: np.ndarray) -> float | None:
    n = adj_c.shape[0]
    if n < 2:
        return None
    upper_tri_mask = np.triu(np.ones(adj_c.shape), k=1)
    upper_tri_values = adj_c[upper_tri_mask == 1]
    return (2 * np.sum(upper_tri_values == 1)) / (n * (n - 1))


def calc_lcc(adj_c: np.ndarray) -> float | None:
    return calc_tcc(to_trans_closure(adj_c))


def calc_lcom5(call_graph: EntityGraph) -> float | None:
    attributes = call_graph.nodes.attributes()
    methods = call_graph.nodes.impl_methods()
    if len(attributes) == 0 or len(methods) < 2:
        return None
    total = 0
    for attribute in attributes:
        incoming = call_graph.edges.incoming(attribute)
        total += len(incoming & methods)
    return (len(methods) - ((1 / len(attributes)) * total)) / (len(methods) - 1)


@dataclass
class CanonicalMetrics:
    lcom1: int
    lcom2: int
    lcom3: int
    lcom4: int
    co: float | None
    tcc: float | None
    lcc: float | None
    lcom5: float | None


def calc_canonical(call_graph: EntityGraph) -> CanonicalMetrics:
    adj_a = calc_graph_a(call_graph)
    adj_b = calc_graph_b(call_graph)
    adj_c = calc_graph_c(call_graph)
    return CanonicalMetrics(
        lcom1=calc_lcom1(adj_a),
        lcom2=calc_lcom2(adj_a),
        lcom3=calc_lcom3(adj_a),
        lcom4=calc_lcom4(adj_b),
        co=calc_co(adj_b),
        tcc=calc_tcc(adj_c),
        lcc=calc_lcc(adj_c),
        lcom5=calc_lcom5(call_graph),
    )


def open_db(db_path: str | PathLike[str]) -> Connection:
    conn = sqlite3.connect(db_path)

    def dict_factory(cursor: Cursor, row: Any):
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    conn.row_factory = dict_factory
    return conn


def split_lines(
    lines: list[str], test_ratio: float, val_ratio: float, seed: int | None
) -> tuple[list[str], list[str], list[str]]:
    # Calculate the number of items in each list
    n = len(lines)
    n_test = math.floor(n * test_ratio)
    n_val = math.floor(n * val_ratio)
    n_train = n - n_test - n_val

    # Shuffle the indices
    indices = list(range(n))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)
    indices_train = sorted(indices[:n_train])
    indices_test = sorted(indices[n_train : n_train + n_test])
    indices_val = sorted(indices[-n_val:])

    # Create new lists
    lines_train = [lines[i] for i in indices_train]
    lines_test = [lines[i] for i in indices_test]
    lines_val = [lines[i] for i in indices_val]
    return lines_train, lines_test, lines_val
