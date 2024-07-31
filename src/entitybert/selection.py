import itertools as it
import math
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import Any, Iterable, Iterator

import pandas as pd
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


@dataclass
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

    def text(self, content: bytes) -> str:
        return content[self.start_byte : self.end_byte].decode()

    def is_file(self) -> bool:
        return self.kind == "File"

    def is_class(self) -> bool:
        return self.kind == "Class"

    def is_attribute(self) -> bool:
        return self.kind == "Field"

    def is_method(self) -> bool:
        return self.kind == "Constructor" or self.kind == "Method"


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
            trees[filename] = tree
        return trees

    def _add_entity(self, entity: EntityDto):
        if entity.parent_id is None and entity.kind != "File":
            raise ValueError(f"Entity (id: {entity.id}) is a root but not a File")
        self._entities[entity.id] = entity
        self._children[entity.parent_id].append(entity.id)

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


class EntityGraph:
    def __init__(self):
        self._incoming: defaultdict[str, set[str]] = defaultdict(set)
        self._outgoing: defaultdict[str, set[str]] = defaultdict(set)

    @staticmethod
    def load_from_db(cursor: Cursor) -> "EntityGraph":
        graph = EntityGraph()
        for dep in _select_deps(cursor):
            graph.add_dep(dep.src, dep.tgt)
        return graph

    def add_dep(self, src: str, tgt: str):
        self._incoming[tgt].add(src)
        self._outgoing[src].add(tgt)

    def only_incoming(self, id: str, ids: set[str]) -> set[str]:
        return self._incoming[id] & ids

    def only_outgoing(self, id: str, ids: set[str]) -> set[str]:
        return self._outgoing[id] & ids

    def to_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for source, targets in self._outgoing.items():
            for target in targets:
                pairs.add((source, target))
        return pairs

    def subgraph(self, ids: set[str]) -> "EntityGraph":
        graph = EntityGraph()
        for src in ids:
            for tgt in self._outgoing[src] & ids:
                graph.add_dep(src, tgt)
        return graph


# This is a blatant copy-paste from "iter_entity_trees"
def iter_entity_graphs(
    files_df: pd.DataFrame, *, pbar: bool
) -> Iterator[tuple[pd.Series, EntityTree, EntityGraph]]:
    bar = tqdm(total=len(files_df), disable=not pbar)
    for db_path, group_df in files_df.groupby("db_path"):
        with open_db(str(db_path)) as conn:
            trees = EntityTree.load_from_db(conn.cursor())
            graph = EntityGraph.load_from_db(conn.cursor())
            for _, row in group_df.iterrows():
                bar.update()
                tree = trees[row["filename"]]  # type: ignore
                yield (row, tree, graph)


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
