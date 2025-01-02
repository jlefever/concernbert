import random
from collections import defaultdict
from typing import Iterator

import pandas as pd
from concernbert import selection


class _ItemLookup[T]:
    def __init__(self):
        self._items: dict[int, list[T]] = defaultdict(list)

    def add_item(self, value: int, item: T):
        self._items[value].append(item)

    def within(self, value_range: range) -> Iterator[T]:
        for value in value_range:
            yield from self._items[value]


class _File:
    def __init__(self, id: int, lloc: int, members: int):
        self.id = id
        self.lloc = lloc
        self.members = members

    def __repr__(self) -> str:
        return f"File(id={self.id}, lloc={self.lloc}, members={self.members})"


class _FileLookup:
    def __init__(self):
        self._files: dict[int, _File] = dict()
        self._by_lloc: _ItemLookup[_File] = _ItemLookup()
        self._by_members: _ItemLookup[_File] = _ItemLookup()

    def add_file(self, file: _File):
        if file.id in self._files:
            raise ValueError("duplicate file id")
        self._files[file.id] = file
        self._by_lloc.add_item(file.lloc, file)
        self._by_members.add_item(file.members, file)

    def rand_file(self) -> _File:
        return random.choice(list(self._files.values()))

    def within(self, lloc_range: range, members_range: range) -> set[_File]:
        lloc = self._by_lloc.within(lloc_range)
        members = self._by_members.within(members_range)
        return set(lloc) & set(members)


class _ProjectLookup:
    def __init__(self):
        self._projects: dict[str, _FileLookup] = defaultdict(_FileLookup)

    def add_file(self, project: str, file: _File):
        self._projects[project].add_file(file)

    def rand_project(self) -> str:
        return random.choice(list(self._projects.keys()))

    def rand_file(self, project: str) -> _File:
        return self._projects[project].rand_file()

    def rand_file_within_range(
        self, project: str, lloc_range: range, members_range: range
    ) -> _File | None:
        files = self._projects[project].within(lloc_range, members_range)
        if len(files) == 0:
            return None
        return random.choice(list(files))

    def rand_file_pair(
        self, lloc_tol: int, members_tol: int
    ) -> tuple[_File, _File] | None:
        a_project = self.rand_project()
        b_project = self.rand_project()
        a_file = self.rand_file(a_project)
        lloc_range = range(max(0, a_file.lloc - lloc_tol), a_file.lloc + lloc_tol + 1)
        members_range = range(
            max(0, a_file.members - members_tol), a_file.members + members_tol + 1
        )
        b_file = self.rand_file_within_range(b_project, lloc_range, members_range)
        if b_file is None:
            return None
        if a_file.id == b_file.id:
            return None
        return (a_file, b_file)

    def sample_n_pairs(
        self, lloc_tol: int, members_tol: int, n: int
    ) -> list[tuple[_File, _File]]:
        ids: set[int] = set()
        pairs: set[tuple[_File, _File]] = set()
        while len(pairs) < n:
            pair = self.rand_file_pair(lloc_tol, members_tol)
            if pair is None:
                continue
            if pair[0].id in ids or pair[1].id in ids:
                continue
            ids.add(pair[0].id)
            ids.add(pair[1].id)
            pairs.add(pair)
        return list(pairs)


def _is_ascii(text: str):
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _augment_files_df(files_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row, tree in selection.iter_entity_trees(files_df, pbar=True):
        cls = tree.standard_class()
        row["is_standard_class"] = cls is not None
        row["members"] = len(tree.children(cls.id)) if cls else None
        row["is_ascii"] = _is_ascii(tree.text())
        row["content"] = tree.text()
        rows.append(row)
    return pd.DataFrame(rows)


def calc_file_ranker_df(
    files_df: pd.DataFrame, *, name: str, seed: int | None, ratio: float, n: int
) -> pd.DataFrame:
    df = _augment_files_df(files_df)
    df = df[df["is_ldl"] & df["is_standard_class"] & df["is_ascii"]]

    # Calculate tolerances
    lloc_tol_raw = df["lloc"].std() * ratio
    lloc_tol = round(lloc_tol_raw)
    print(f"Using a LLOC tol of {lloc_tol} (rounded from {lloc_tol_raw})")
    members_tol_raw = df["members"].std() * ratio
    members_tol = round(members_tol_raw)
    print(f"Using a members tol of {members_tol} (rounded from {members_tol_raw})")

    # Build ProjectLookup
    if seed:
        random.seed(seed)
    project_lookup = _ProjectLookup()
    for ix, row in df.iterrows():
        lloc = row["lloc"]
        members = int(row["members"])
        project_lookup.add_file(row["db_path"], _File(int(ix), lloc, members))  # type: ignore

    # Generate pairs
    random.seed(seed)
    pairs = project_lookup.sample_n_pairs(
        lloc_tol=lloc_tol, members_tol=members_tol, n=n
    )

    # Create DataFrame
    out_rows = []
    for position, (file_a, file_b) in enumerate(pairs):
        row_a = df.loc[file_a.id]
        row_b = df.loc[file_b.id]
        out_rows.append(
            {
                "sequence": name,
                "position": position,
                "project_a": row_a["db_path"],
                "project_b": row_b["db_path"],
                "filename_a": row_a["filename"],
                "filename_b": row_b["filename"],
                "content_a": row_a["content"],
                "content_b": row_b["content"],
            }
        )
    return pd.DataFrame.from_records(out_rows, index="position")


def load_files_from_seq_df(
    seq_df: pd.DataFrame, *, max_pos: int | None
) -> pd.DataFrame:
    df = seq_df
    if max_pos is not None:
        df = df[df["position"] <= max_pos]
    a_df = df[["project_a", "filename_a"]].drop_duplicates()
    b_df = df[["project_b", "filename_b"]].drop_duplicates()
    a_df = a_df.rename(columns={"project_a": "db_path", "filename_a": "filename"})
    b_df = b_df.rename(columns={"project_b": "db_path", "filename_b": "filename"})
    mentioned_df = pd.concat([a_df, b_df], ignore_index=True).drop_duplicates()
    mentioned_df = mentioned_df.sort_values(["db_path", "filename"]).reset_index()
    db_paths = sorted(set(mentioned_df["db_path"]))
    files_df = selection.load_multi_files_df(db_paths)
    merged_df = mentioned_df.merge(files_df, how="left", on=["db_path", "filename"])
    if merged_df.isnull().values.any():
        raise RuntimeError("failed to load all files")
    return merged_df.drop(columns="index")
