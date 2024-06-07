import logging
import math
import random
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
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
    for row in cursor.fetchall():
        yield _FileDto(**row)


_SELECT_DEPS = """
    SELECT DISTINCT SF.filename AS src, TF.filename AS tgt
    FROM deps D
    JOIN temp.filenames SF ON SF.entity_id = D.src
    JOIN temp.filenames TF ON TF.entity_id = D.tgt
    ORDER BY SF.filename, TF.filename
"""


@dataclass
class _DepDto:
    src: str
    tgt: str


def _select_deps(cursor: Cursor) -> Iterable[_DepDto]:
    cursor.execute(_SELECT_DEPS)
    for row in cursor.fetchall():
        yield _DepDto(**row)


_SELECT_CHANGES = """
    SELECT DISTINCT F.filename, HEX(C.commit_id) AS commit_id
    FROM changes C
    JOIN temp.filenames F ON F.simple_id = C.simple_id
"""


@dataclass
class _ChangeDto:
    filename: str
    commit_id: str


def _select_changes(cursor: Cursor) -> Iterable[_ChangeDto]:
    cursor.execute(_SELECT_CHANGES)
    for row in cursor.fetchall():
        yield _ChangeDto(**row)


_SELECT_ENTITIES = """
    SELECT
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
    WHERE F.filename = ?
"""


@dataclass
class _EntityDto:
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


def _select_entities(cursor: Cursor, filename: str) -> Iterable[_EntityDto]:
    cursor.execute(_SELECT_ENTITIES, (filename,))
    for row in cursor.fetchall():
        yield _EntityDto(**row)


_SELECT_CONTENTS = """
    SELECT C.content
    FROM entities E
    JOIN contents C ON C.content_id = E.content_id
    WHERE E.kind = 'File' AND E.name = ?
"""


@dataclass
class _ContentDto:
    content: bytes


def _select_contents(cursor: Cursor, filename: str) -> Iterable[_ContentDto]:
    cursor.execute(_SELECT_CONTENTS, (filename,))
    for row in cursor.fetchall():
        row["content"] = row["content"].encode()
        yield _ContentDto(**row)


_INVALID_ENDINGS = [
    "Test.java",
    "Tests.java",
    "Example.java",
    "Examples.java",
    "Benchmark.java",
    "Benchmarks.java",
    "-info.java",
]

_INVALID_SEGMENTS = set(
    [
        "test",
        "tests",
        "integration-test",
        "integration-tests",
        "testkit",
        "gen",
        "generated",
        "example",
        "examples",
        "benchmark",
        "benchmarks",
    ]
)


def is_filename_valid(filename: str) -> bool:
    if any(filename.endswith(e) for e in _INVALID_ENDINGS):
        return False
    segments = filename.lower().split("/")
    return not any(s in _INVALID_SEGMENTS for s in segments)


class _FileGraph:
    def __init__(self):
        self._files: dict[str, _FileDto] = dict()
        self._incoming: defaultdict[str, set[str]] = defaultdict(set)
        self._outgoing: defaultdict[str, set[str]] = defaultdict(set)
        self._commits: defaultdict[str, set[str]] = defaultdict(set)

    @staticmethod
    def load_from_db(cursor: Cursor) -> "_FileGraph":
        graph = _FileGraph()
        for file in _select_files(cursor):
            graph.add_file(file)
        for dep in _select_deps(cursor):
            graph.add_dep(dep.src, dep.tgt)
        for change in _select_changes(cursor):
            graph.add_change(change.filename, change.commit_id)
        return graph

    def add_file(self, file: _FileDto):
        self._files[file.filename] = file

    def add_dep(self, src: str, tgt: str):
        self._incoming[tgt].add(src)
        self._outgoing[src].add(tgt)

    def add_change(self, file: str, commit: str):
        self._commits[file].add(commit)

    def files(self) -> list[str]:
        return list(self._files)

    def commits_of(self, file: str) -> set[str]:
        return self._commits[file]

    def in_deps_of(self, file: str) -> set[str]:
        return self._incoming[file]

    def out_deps_of(self, file: str) -> set[str]:
        return self._outgoing[file]

    def non_deps_of(self, file: str) -> set[str]:
        return self._files.keys() - self.in_deps_of(file) - self.out_deps_of(file)

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
        row["revisions"] = len(self.commits_of(file))
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
        self._entities: dict[str, _EntityDto] = dict()
        self._children: defaultdict[str | None, list[str]] = defaultdict(list)

    @staticmethod
    def load_from_db(cursor: Cursor, filename: str) -> "EntityTree":
        contents = list(_select_contents(cursor, filename))
        if len(contents) != 1:
            raise RuntimeError(
                f"found {len(contents)} contents. Expected 1. ({filename})"
            )
        tree = EntityTree(filename, contents[0].content)
        for entity in _select_entities(cursor, filename):
            tree._add_entity(entity)
        return tree

    def _add_entity(self, entity: _EntityDto):
        if entity.parent_id is None and entity.kind != "File":
            raise ValueError(f"Entity (id: {entity.id}) is a root but not a File")
        self._entities[entity.id] = entity
        self._children[entity.parent_id].append(entity.id)

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


def open_db(db_path: str | PathLike[str]) -> Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_CREATE_TEMP_TABLES)

    def dict_factory(cursor: Cursor, row: Any):
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    conn.row_factory = dict_factory
    return conn


def load_files_df(db_path: str) -> pd.DataFrame | None:
    conn = None
    try:
        graph = _FileGraph.load_from_db(open_db(db_path).cursor())
        return graph.to_files_df(db_path)
    except sqlite3.OperationalError:
        return None
    finally:
        if conn:
            conn.close()


def load_many_files_df(db_paths: list[str], *, pbar: bool = False) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    skipped_paths: list[str] = []
    for db_path in tqdm(db_paths, disable=not pbar):
        files_df = load_files_df(db_path)
        if files_df is not None:
            dfs.append(files_df)
        else:
            skipped_paths.append(db_path)
    for path in skipped_paths:
        logger.warn(f"Could not load file data from {path}")
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
        group_df = df[df["db_path"] == db_path]
        filenames: list[str] = sorted(set((group_df["filename"])))  # type: ignore
        for filename in filenames:
            bar.update()
            try:
                tree = EntityTree.load_from_db(conn.cursor(), filename)
                dfs.append(tree.to_entities_df(db_path))
            except RuntimeError as e:
                logger.warn(f"Skipping file in {db_path} because of error: {e}")
                continue
    return pd.concat(dfs, ignore_index=True)  # type: ignore


def list_files(cursor: Cursor) -> list[str]:
    return [d.filename for d in _select_files(cursor)]


def list_valid_files(cursor: Cursor) -> list[str]:
    return [f for f in list_files(cursor) if is_filename_valid(f)]


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
