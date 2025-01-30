from dataclasses import dataclass
import itertools as it

import numpy as np

from concernbert.semantic import find_entity_docs
from concernbert.embeddings import load_caching_embedder
from concernbert.metrics import to_aad


@dataclass
class CdResult:
    inter_cd: float
    intra_cd: float
    groups: dict[str, float]
    num_entities: int


class CdCalculator:
    def __init__(self, model: str, cache_dir: str, batch_size: int = 24):
        self._embedder = load_caching_embedder(model, cache_dir, batch_size)
    
    def calc_cd(self, source: str, *, pbar: bool = False) -> CdResult:
        res = find_entity_docs(source)
        texts = list(it.chain(*res.values()))
        embeddings = self._embedder.embed(texts, pbar=pbar)
        groups: dict[str, float] = dict()
        for group_name, group_texts in res.items():
            group_embeddings = [embeddings[t] for t in group_texts]
            groups[group_name] = to_aad(np.array(group_embeddings))
        inter_cd = float(np.mean(list(groups.values())))
        intra_cd = to_aad(np.array(list(embeddings.values())))
        return CdResult(inter_cd, intra_cd, groups, len(texts))
