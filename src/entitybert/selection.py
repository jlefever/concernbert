import itertools as it
import logging
import math
import random
import sqlite3
import subprocess as sp
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from io import StringIO
from os import PathLike
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CREATE_TEMP_TABLES = """
    CREATE TEMP TABLE temp.ancestors AS
    WITH RECURSIVE ancestors (entity_id, ancestor_id) AS
    (
        SELECT E.id AS entity_id, E.id AS ancestor_id
        FROM entities E

        UNION ALL

        SELECT E.id AS entity_id, A.ancestor_id
        FROM ancestors A
        JOIN entities E ON A.entity_id = E.parent_id
    )
    SELECT * FROM ancestors;

    CREATE TEMP TABLE temp.filenames AS
    SELECT
        E.id AS entity_id,
        E.simple_id AS simple_id,
        FE.id AS file_id,
        FE.name AS filename
    FROM entities E
    JOIN temp.ancestors A ON A.entity_id = E.id
    JOIN entities FE ON FE.id = A.ancestor_id
    WHERE FE.parent_id IS NULL;

    CREATE TEMP TABLE IF NOT EXISTS temp.levels AS
    WITH RECURSIVE levels (entity_id, level) AS
    (
        SELECT E.id AS entity_id, 0 as level
        FROM entities E
        WHERE E.parent_id IS NULL

        UNION ALL

        SELECT E.id AS entity_id, L.level + 1
        FROM entities E, levels L
        WHERE E.parent_id = L.entity_id
    )
    SELECT * FROM levels;
"""

_SELECT_FILES = """
    SELECT
        F.filename,
        MAX(E.end_row) + 1 AS loc,
        COUNT(E.id) AS n_entities
    FROM entities E
    JOIN temp.filenames F ON F.entity_id = E.id
    GROUP BY F.filename
"""


@dataclass
class _FileDto:
    filename: str
    loc: int
    n_entities: int


def _select_files(cursor: Cursor) -> Iterable[_FileDto]:
    cursor.execute(_SELECT_FILES)
    yield from (_FileDto(**r) for r in cursor.fetchall())


_SELECT_FILE_DEPS = """
    SELECT DISTINCT SF.filename AS src, TF.filename AS tgt
    FROM deps D
    JOIN temp.filenames SF ON SF.entity_id = D.src
    JOIN temp.filenames TF ON TF.entity_id = D.tgt
    WHERE SF.filename <> TF.filename
    ORDER BY SF.filename, TF.filename
"""


@dataclass
class _FileDepDto:
    src: str
    tgt: str


def _select_file_deps(cursor: Cursor) -> Iterable[_FileDepDto]:
    cursor.execute(_SELECT_FILE_DEPS)
    yield from (_FileDepDto(**r) for r in cursor.fetchall())


_SELECT_FILE_CHANGES = """
    SELECT DISTINCT F.filename, HEX(C.commit_id) AS commit_id
    FROM changes C
    JOIN temp.filenames F ON F.simple_id = C.simple_id
"""


@dataclass
class _FileChangeDto:
    filename: str
    commit_id: str


def _select_file_changes(cursor: Cursor) -> Iterable[_FileChangeDto]:
    cursor.execute(_SELECT_FILE_CHANGES)
    yield from (_FileChangeDto(**r) for r in cursor.fetchall())


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
        "test",
        "testkit",
        "tests",
        "tutorial",
        "tutorials",
    ]
)


def is_filename_valid(filename: str) -> bool:
    if any(filename.endswith(e) for e in _INVALID_ENDINGS):
        return False
    segments = filename.lower().split("/")
    return not any(s in _INVALID_SEGMENTS for s in segments)


class FileGraph:
    def __init__(self):
        self._files: dict[str, _FileDto] = dict()
        self._incoming: defaultdict[str, set[str]] = defaultdict(set)
        self._outgoing: defaultdict[str, set[str]] = defaultdict(set)
        self._file_to_commits: defaultdict[str, set[str]] = defaultdict(set)
        self._commit_to_files: defaultdict[str, set[str]] = defaultdict(set)

    @staticmethod
    def load_from_db(cursor: Cursor) -> "FileGraph":
        graph = FileGraph()
        for file in _select_files(cursor):
            graph.add_file(file)
        for dep in _select_file_deps(cursor):
            graph.add_dep(dep.src, dep.tgt)
        for change in _select_file_changes(cursor):
            graph.add_change(change.filename, change.commit_id)
        return graph

    def add_file(self, file: _FileDto):
        self._files[file.filename] = file

    def add_dep(self, src: str, tgt: str):
        self._incoming[tgt].add(src)
        self._outgoing[src].add(tgt)

    def add_change(self, file: str, commit: str):
        self._file_to_commits[file].add(commit)
        self._commit_to_files[commit].add(file)

    def files(self) -> list[str]:
        return list(self._files)

    def files_of(self, commit: str) -> set[str]:
        return self._commit_to_files[commit]

    def commits_of(self, file: str) -> set[str]:
        return self._file_to_commits[file]

    def in_deps_of(self, file: str) -> set[str]:
        return self._incoming[file]

    def out_deps_of(self, file: str) -> set[str]:
        return self._outgoing[file]

    def non_deps_of(self, file: str) -> set[str]:
        return self._files.keys() - self.in_deps_of(file) - self.out_deps_of(file)

    @cache
    def cochange_counts(self, file: str) -> Counter[str]:
        counter = Counter()
        for commit in self.commits_of(file):
            counter.update(self.files_of(commit))
        del counter[file]
        return counter

    @cache
    def nontrivial_cochange_counts(self, file: str) -> Counter[str]:
        return Counter(
            {k: c - 1 for k, c in self.cochange_counts(file).items() if c > 1}
        )

    def cochange(self, file_a: str, file_b: str) -> int:
        return len(self.commits_of(file_a).intersection(self.commits_of(file_b)))

    def cochange_many(self, file_a: str, file_bs: set[str]) -> int:
        return sum(self.cochange(file_a, b) for b in file_bs)

    def cochange_fan_in(self, file: str) -> int:
        return self.cochange_many(file, self.in_deps_of(file))

    def cochange_fan_out(self, file: str) -> int:
        return self.cochange_many(file, self.out_deps_of(file))

    def cochange_cross(self, file: str) -> int:
        others = self.out_deps_of(file).union(self.in_deps_of(file))
        return self.cochange_many(file, others)

    def cochange_no_dep(self, file: str) -> int:
        return self.cochange_many(file, self.non_deps_of(file))

    def to_file_row(self, db_path: str | None, file: str) -> dict[str, Any]:
        file_dto = self._files[file]
        row: dict[str, Any] = {}
        row["db_path"] = db_path
        row["filename"] = file_dto.filename
        row["loc"] = file_dto.loc
        row["n_entities"] = file_dto.n_entities
        row["is_name_valid"] = is_filename_valid(file_dto.filename)
        row["fan_in"] = len(self.in_deps_of(file))
        row["fan_out"] = len(self.out_deps_of(file))
        row["commits"] = len(self.commits_of(file))
        row["CC (in)"] = self.cochange_fan_in(file)
        row["CC (out)"] = self.cochange_fan_out(file)
        row["CC (cross)"] = self.cochange_cross(file)
        row["CC (no dep)"] = self.cochange_no_dep(file)
        return row

    def to_files_df(self, db_path: str) -> pd.DataFrame:
        rows = [self.to_file_row(db_path, f) for f in self.files()]
        return pd.DataFrame.from_records(rows)  # type: ignore


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

    def loc(self) -> int:
        return self._content.count(0x0A) + 1

    def is_leaf(self, id: str) -> bool:
        "Returns true if this entity has no children"
        return len(self._children[id]) == 0

    def leaf_children(self, id: str) -> list[str]:
        "Returns a list of leaf children for this entity"
        return [c for c in self._children[id] if self.is_leaf(c)]

    def leaf_siblings(self) -> list[list[str]]:
        return [self.leaf_children(id) for id in self._entities]

    def nontrivial_leaf_siblings(self) -> list[list[str]]:
        return [s for s in self.leaf_siblings() if len(s) > 1]

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


def open_db(db_path: str | PathLike[str]) -> Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_CREATE_TEMP_TABLES)  # type: ignore

    def dict_factory(cursor: Cursor, row: Any):
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    conn.row_factory = dict_factory
    return conn


def load_files_df(db_paths: list[str], *, pbar: bool = False) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for db_path in tqdm(db_paths, disable=not pbar):
        graph = FileGraph.load_from_db(open_db(db_path).cursor())
        dfs.append(graph.to_files_df(db_path))
    return pd.concat(dfs, ignore_index=True)  # type: ignore


def filter_files_df(
    df: pd.DataFrame,
    global_quantile: float,
    local_quantile: float,
    keep_invalid_names: bool,
) -> pd.DataFrame:
    if not keep_invalid_names:
        df = df[df["is_name_valid"]]
    columns = ["loc", "n_entities", "CC (in)", "CC (out)", "CC (cross)", "CC (no dep)"]
    df = df[(df[columns] != 0).all(axis=1)]  # type: ignore
    global_thresholds = df[columns].quantile(global_quantile)  # type: ignore
    local_thresholds = df.groupby("db_path")[columns].quantile(local_quantile)  # type: ignore
    df = df[
        (df[columns] <= global_thresholds).all(axis=1)  # type: ignore
        & (
            ~df.apply(  # type: ignore
                lambda x: (x[columns] > local_thresholds.loc[x["db_path"]]).any(),  # type: ignore
                axis=1,
            )
        )
    ]
    return df


def extract_entities_df(df: pd.DataFrame, *, pbar: bool = False) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    db_paths: list[str] = sorted(set(df["db_path"]))  # type: ignore
    bar = tqdm(total=len(df), disable=not pbar)
    for db_path in db_paths:
        conn = open_db(db_path)
        trees = EntityTree.load_from_db(conn.cursor())
        group_df = df[df["db_path"] == db_path]
        filenames: list[str] = sorted(set((group_df["filename"])))  # type: ignore
        for filename in filenames:
            bar.update()
            tree = trees[filename]
            dfs.append(tree.to_entities_df(db_path))
    return pd.concat(dfs, ignore_index=True)  # type: ignore


def run_scc(trees: Iterable[EntityTree]) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as temp_dir:
        for tree in trees:
            path = Path(temp_dir, tree.filename())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(tree.text())
        args = ["scc", "--by-file", "--format=csv"]
        csv = sp.run(args, capture_output=True, cwd=temp_dir).stdout.decode()
        df = pd.read_csv(StringIO(csv))
        df = df.drop(columns=["Filename"]).rename(columns={"Provider": "Filename"})
        return df.set_index("Filename")


def prepare_file_ranker_df(db_path: str) -> pd.DataFrame:
    conn = open_db(db_path)
    trees = EntityTree.load_from_db(conn.cursor())
    graph = FileGraph.load_from_db(conn.cursor())
    scc_df = run_scc(trees.values())
    rows: list[dict[str, Any]] = []
    for filename, tree in trees.items():
        if not is_filename_valid(filename):
            continue
        groups = tree.nontrivial_leaf_siblings()
        if len(groups) != 1:
            continue
        siblings = groups[0]
        entities = [tree[id] for id in siblings]
        parent_id = entities[0].parent_id
        if parent_id is None or tree[parent_id].kind != "Class":
            continue
        row: dict[str, Any] = dict()
        row["filename"] = filename
        row["loc"] = scc_df.loc[filename]["Lines"]
        row["lloc"] = scc_df.loc[filename]["Code"]
        row["entities"] = len(entities)
        row["commits"] = len(graph.commits_of(filename))
        row["content"] = tree.text()
        rows.append(row)
    return pd.DataFrame.from_records(rows)


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
