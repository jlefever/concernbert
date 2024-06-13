import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from statistics import mean
from typing import Hashable, Iterator, TypeVar

import pandas as pd
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

_T = TypeVar("_T")
_Batch = list[_T]
_Epoch = list[_Batch[_T]]


@dataclass
class Point[V, L]:
    value: V
    label: L


class _ValueSampler[V]:
    def __init__(self):
        self._values: list[V] = []

    def insert(self, value: V):
        self._values.append(value)

    def sample(self) -> V | None:
        if len(self._values) == 0:
            return None
        return self._values.pop(random.randint(0, len(self) - 1))

    def __len__(self) -> int:
        return len(self._values)


@dataclass
class _BatchSamplerArgs:
    min_labels: int
    max_points_per_label: int


class _BatchSampler[V, L]:
    def __init__(self):
        self._samplers: dict[L, _ValueSampler[V]] = defaultdict(_ValueSampler)

    def insert(self, value: V, label: L):
        self._samplers[label].insert(value)

    def sample(self, args: _BatchSamplerArgs) -> _Batch[Point[V, L]] | None:
        if args.min_labels < 2:
            raise ValueError("`min_labels` must be at least 2")
        if args.max_points_per_label < 2:
            raise ValueError("`max_points_per_label` must be at least 2")
        n = args.min_labels * args.max_points_per_label
        batch: list[Point[V, L]] = []
        labels = list(self._samplers.keys())
        random.shuffle(labels)
        for label in labels:
            if len(batch) == n:
                return batch
            cluster = self._samplers[label]
            if len(cluster) < 2:
                self._samplers.pop(label)
                continue
            remaining = n - len(batch)
            n_samples = min(args.max_points_per_label, len(cluster), remaining)
            # Skip this label if adding this number of samples to the batch will
            # leave only one slot left to be filled. It would be impossible to
            # fill this slot because we need at least two samples from each
            # label.
            if remaining - n_samples != 1:
                for _ in range(n_samples):
                    batch.append(Point(cluster.sample(), label))  # type: ignore
        return batch if len(batch) == n else None


class _MultiBatchSampler[V, L]:
    def __init__(self) -> None:
        self._samplers: dict[str, _BatchSampler[V, L]] = defaultdict(_BatchSampler)

    def insert(self, value: V, label: L, name: str):
        self._samplers[name].insert(value, label)

    def sample(self, args: _BatchSamplerArgs) -> _Batch[Point[V, L]] | None:
        names = list(self._samplers.keys())
        random.shuffle(names)
        for name in names:
            if (sample := self._samplers[name].sample(args)) is not None:
                return sample
        return None

    def to_epoch(self, args: _BatchSamplerArgs) -> _Epoch[Point[V, L]]:
        epoch: _Epoch[Point[V, L]] = []
        while (batch := self.sample(args)) is not None:
            epoch.append(batch)
        return epoch


class _IndexMap[T: Hashable]:
    def __init__(self):
        self._id_to_index: dict[T, int] = dict()

    def index(self, id: T) -> int:
        if (index := self._id_to_index.get(id)) is not None:
            return index
        index = len(self._id_to_index)
        self._id_to_index[id] = index
        return index


class MyInputExample:
    def __init__(self, ix: int, text: str, label: int) -> None:
        self.ix = ix
        self.texts = [text]
        self.label = label

    def __str__(self) -> str:
        first_line = self.texts[0].split("\n")[0]
        return f'MyInputExample({self.ix}, "{first_line}", {self.label})'

    def __repr__(self) -> str:
        return f'MyInputExample({self.ix}, """{self.texts[0]}""", {self.label})'


class MySampler(Sampler[list[int]]):
    def __init__(self, epochs: list[_Epoch[Point[int, int]]]):
        # Assumes:
        # - There is at least one epoch
        # - All epochs have at least one batch
        # - All batches are the same size
        batch_size = len(epochs[0][0])
        self._epochs = epochs
        self._n_epochs = len(epochs)
        self._n_batches_per_epoch = min(len(e) for e in self._epochs)
        self._n_points_per_batch = batch_size
        self._curr_epoch = 0

    def n_epochs(self) -> int:
        """The number of epochs in this sampler"""
        return self._n_epochs

    def n_batches(self) -> int:
        """The number of batches across all epochs in this sampler"""
        return self.n_epochs() * self.n_batches_per_epoch()

    def n_points(self) -> int:
        """The number of points across all epochs in this sampler"""
        return self.n_points_per_epoch() * self.n_epochs()

    def n_batches_per_epoch(self) -> int:
        """The number of batches per epoch in this sampler"""
        return self._n_batches_per_epoch

    def n_points_per_epoch(self) -> int:
        """The number of points per epoch in this sampler"""
        return self.n_points_per_batch() * self.n_batches_per_epoch()

    def n_points_per_batch(self) -> int:
        """The number of points per batch per epoch in this sampler (i.e. batch size)"""
        return self._n_points_per_batch

    @cache
    def avg_labels_per_batch(self) -> float:
        n_labels: list[int] = []
        for epoch in self._epochs:
            for batch in epoch:
                n_labels.append(len(set(p.label for p in batch)))
        return mean(n_labels)

    @cache
    def avg_points_per_label_per_batch(self) -> float:
        n_points: list[int] = []
        for epoch in self._epochs:
            for batch in epoch:
                n_points.extend(Counter(p.label for p in batch).values())
        return mean(n_points)

    @cache
    def summary(self) -> str:
        lines: list[str] = [
            f"# of Epochs:  {self.n_epochs()}",
            f"# of Batches: {self.n_batches()}",
            f"# of Points:  {self.n_points()}",
            f"# of Batches per Epoch: {self.n_batches_per_epoch()}",
            f"# of Points per Epoch:  {self.n_points_per_epoch()}",
            f"# of Points per Batch:  {self.n_points_per_batch()}",
            f"Avg. # of Labels per Batch:           {self.avg_labels_per_batch()}",
            f"Avg. # of Points per Label per Batch: {self.avg_points_per_label_per_batch()}",
        ]
        return "\n".join(lines)

    def __iter__(self) -> Iterator[list[int]]:
        if self._curr_epoch > self.n_epochs():
            raise RuntimeError("MySampler.__iter__ called too many times")
        batches = self._epochs[self._curr_epoch]
        self._curr_epoch += 1
        yield from ([p.value for p in b] for b in batches[: self.n_batches_per_epoch()])

    def __len__(self) -> int:
        return self.n_batches_per_epoch()


@dataclass
class SamplerArgs:
    seed: int
    epochs: int
    min_labels: int
    max_points_per_label: int


class MyDataset(Dataset[MyInputExample]):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._labels = _IndexMap[str]()

    def sampler(self, args: SamplerArgs) -> MySampler:
        random.seed(args.seed)
        epochs: list[_Epoch[Point[int, int]]] = []
        for _ in range(args.epochs):
            sampler = _MultiBatchSampler[int, int]()
            for ix, row in tqdm(self._df.iterrows()):  # type: ignore
                label = self._labels.index(str(row["parent_id"]))  # type: ignore
                db_path = str(row["db_path"])  # type: ignore
                sampler.insert(int(ix), label, db_path)  # type: ignore
            batch_args = _BatchSamplerArgs(args.min_labels, args.max_points_per_label)
            epochs.append(sampler.to_epoch(batch_args))
        return MySampler(epochs)

    def __getitem__(self, ix: int) -> MyInputExample:
        row = self._df.loc[ix]  # type: ignore
        text: str = str(row["content"])  # type: ignore
        label: int = self._labels.index(str(row["parent_id"]))  # type: ignore
        return MyInputExample(ix, text, label)

    def __len__(self) -> int:
        return len(self._df)
