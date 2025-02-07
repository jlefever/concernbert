from dataclasses import dataclass

import numpy as np
import scipy as sp

from concernbert.semantic import find_entity_docs, EntityDoc
from concernbert.embeddings import load_caching_embedder
from concernbert.metrics import to_aad


JOHNSONSU_PARAMS = (
    1.389116411214,
    2.188164765298982,
    1.5768267780674932,
    0.4280317807942762,
)


def estimate_percentile(value: float) -> float:
    shape1, shape2, loc, scale = JOHNSONSU_PARAMS
    return float(sp.stats.johnsonsu.cdf(value, shape1, shape2, loc, scale) * 100)


@dataclass
class CdResult:
    inter_cd: float
    intra_cd: float
    groups: list[float]
    num_entities: int
    embeddings: list[list[np.ndarray]]


class CdCalculator:
    def __init__(self, model: str, cache_dir: str, batch_size: int = 24):
        self._embedder = load_caching_embedder(model, cache_dir, batch_size)

    def calc_cd(self, source: str, *, pbar: bool = False) -> CdResult:
        groups = find_entity_docs(source)
        return self.calc_cd_from_docs(groups, pbar=pbar)

    def calc_cd_from_docs(
        self, groups: list[list[EntityDoc]], *, pbar: bool = False
    ) -> CdResult:
        texts = [doc.text for group in groups for doc in group]
        all_embeddings = self._embedder.embed(texts, pbar=pbar)
        group_cds: list[float] = list()
        embeddings: list[list[np.ndarray]] = []
        for group in groups:
            group_texts = [doc.text for doc in group]
            group_embeddings = [all_embeddings[t] for t in group_texts]
            group_cds.append(to_aad(np.array(group_embeddings)))
            embeddings.append(group_embeddings)
        inter_cd = float(np.mean(list(group_cds)))
        intra_cd = to_aad(np.array(list(all_embeddings.values())))
        return CdResult(
            inter_cd,
            intra_cd,
            group_cds,
            len(texts),
            embeddings,
        )
