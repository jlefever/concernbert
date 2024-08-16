import enum
import itertools as it
import logging
from collections import defaultdict
from dataclasses import dataclass

import gensim
import nltk
import numpy as np
import tree_sitter_languages
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tree_sitter import Node

nltk.download("stopwords")
nltk.download("punkt")
logging.getLogger("gensim").setLevel(logging.ERROR)


_JAVA_LANGUAGE = tree_sitter_languages.get_language("java")
_JAVA_PARSER = tree_sitter_languages.get_parser("java")
_JAVA_QUERY = _JAVA_LANGUAGE.query(
    """
    (class_declaration
        name: (identifier) @identifier) @class
    (record_declaration
        name: (identifier) @identifier) @record
    (enum_declaration
        name: (identifier) @identifier) @enum
    (interface_declaration
        name: (identifier) @identifier) @interface
    (annotation_type_declaration
        name: (identifier) @identifier) @annotation
    (method_declaration
        name: (identifier) @identifier) @method
    (constructor_declaration
        name: (identifier) @identifier) @constructor
    (field_declaration
        declarator: (variable_declarator
            name: (identifier) @identifier)) @field
    (line_comment) @line_comment
    (block_comment) @block_comment
"""
)


class _CaptureKind(enum.Enum):
    """Different capture kinds found in `JAVA_QUERY`."""

    # Types
    ANNOTATION = 1
    CLASS = 2
    ENUM = 3
    INTERFACE = 4
    RECORD = 5

    # Functions
    CONSTRUCTOR = 6
    METHOD = 7

    # Fields
    FIELD = 8

    # Block comment (includes both normal block comments and doc comments)
    LINE_COMMENT = 9
    BLOCK_COMMENT = 10

    # Identifier
    IDENTIFIER = 11

    def is_function(self) -> bool:
        return self == _CaptureKind.CONSTRUCTOR or self == _CaptureKind.METHOD

    def is_comment(self) -> bool:
        return self == _CaptureKind.LINE_COMMENT or self == _CaptureKind.BLOCK_COMMENT


@dataclass(frozen=True)
class _Capture:
    node: Node
    kind: _CaptureKind


def _determine_lineage(node_a: Node, node_b: Node) -> tuple[Node, Node] | None:
    """
    Calculate the ancestor-descendant relationship between two nodes.

    Returns the tuple (ancestor, descendant). If the two nodes do not have
    a strict ancestor-descendant relationship, then None is returned.
    """
    a0, a1 = node_a.byte_range
    b0, b1 = node_b.byte_range
    # Check if completely overlapping
    if a0 == b0 and a1 == b1:
        # TODO: Throw exception instead?
        return None
    # Check if there is no overlap
    if a1 <= b0 or b1 <= a0:
        return None
    # Check if a completely surrounds b
    if a0 <= b0 and b1 <= a1:
        return node_a, node_b
    # Check if b completely surrounds a
    if b0 <= a0 and a1 <= b1:
        return node_b, node_a
    # There is some amount of overlap but no clear hierarchy
    # TODO: Throw exception instead?
    return None


def _calculate_parents_dict(nodes: list[Node]) -> dict[Node, Node | None]:
    """Given a list of nodes, return a child-to-parent mapping."""
    parents: dict[Node, Node | None] = {n: None for n in nodes}
    for node_a, node_b in it.combinations(nodes, 2):
        if (lineage := _determine_lineage(node_a, node_b)) is None:
            continue
        ancestor, descendant = lineage
        current_parent = parents.get(descendant, None)
        if current_parent is None:
            parents[descendant] = ancestor
            continue
        if (lineage := _determine_lineage(current_parent, ancestor)) is None:
            raise RuntimeError(
                "Two nodes with a common descendent have a non-overlapping byte range"
            )
        parents[descendant] = lineage[1]
    return parents


def _to_children_dict(
    parents: dict[Node, Node | None],
) -> dict[Node | None, list[Node]]:
    """Return a parent-to-children mapping given a child-to-parent mapping."""
    children: dict[Node | None, list[Node]] = defaultdict(list)
    for child, parent in parents.items():
        children[parent].append(child)
    return dict(children)


def _get_roots(children: dict[Node | None, list[Node]]) -> list[Node]:
    return children.get(None, [])


def _find_captures(content_bytes: bytes) -> dict[int, _Capture]:
    captures: dict[int, _Capture] = {}
    tree = _JAVA_PARSER.parse(content_bytes)
    for node, capture_name in _JAVA_QUERY.captures(tree.root_node):
        captures[node.id] = _Capture(node, _CaptureKind[capture_name.upper()])
    return captures


def join_singles(terms: list[str]) -> list[str]:
    ret = []
    joined_term = []
    for t in terms:
        if len(t) == 1:
            joined_term.append(t[0])
        elif len(t) > 1:
            if len(joined_term) > 0:
                ret.append("".join(joined_term))
                joined_term = []
            ret.append(t)
    if len(joined_term) > 0:
        ret.append("".join(joined_term))
    return ret


def split_camel(name: str) -> list[str]:
    if name.isupper():
        return [name.lower()]
    indices = [i for i, x in enumerate(name) if x.isupper() or x.isnumeric()]
    indices = [0] + indices + [len(name)]
    return join_singles([name[a:b].lower() for a, b in it.pairwise(indices)])


def split_identifier(name: str) -> list[str]:
    by_spaces = name.split(" ")
    by_underscores = it.chain(*(z.split("_") for z in by_spaces))
    return list(it.chain(*(split_camel(z) for z in by_underscores)))


def preprocess(text) -> list[str]:
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return [stemmer.stem(i) for i in tokens]


def find_documents(content: str) -> list[list[str]]:
    content_bytes = content.encode()
    captures = _find_captures(content_bytes)
    nodes = [c.node for c in captures.values()]
    parents = _calculate_parents_dict(nodes)
    children = _to_children_dict(parents)
    roots = _get_roots(children)
    root_classes = [r for r in roots if captures[r.id].kind == _CaptureKind.CLASS]
    if len(root_classes) != 1:
        return []
    members = [m for m in children[root_classes[0]]]
    members.sort(key=lambda m: m.byte_range)
    docs: list[list[str]] = []
    for i, member in enumerate(members):
        doc: list[str] = []
        capture = captures[member.id]
        if not capture.kind.is_function():
            continue
        for child in children[member]:
            if captures[child.id].kind == _CaptureKind.IDENTIFIER:
                stemmer = PorterStemmer()
                doc += [stemmer.stem(t) for t in split_identifier(child.text.decode())]
            if captures[child.id].kind.is_comment():
                doc += preprocess(child.text.decode())
        if i - 1 >= 0 and captures[members[i - 1].id].kind.is_comment():
            doc += preprocess(members[i - 1].text.decode())
        docs.append(doc)
    return docs


# def compute_coherence_values(dictionary, corpus, texts, *, start, limit, step):
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.LsiModel(
#             corpus, num_topics=num_topics, id2word=dictionary
#         )
#         model_list.append(model)
#         model = CoherenceModel(
#             model=model, texts=texts, dictionary=dictionary, coherence="c_v"
#         )
#         coherence_values.append(model.get_coherence())
#     return model_list, coherence_values


# def calc_sim_matrix_coherence(docs, *, start=2, limit=10, step=2):
#     dictionary = corpora.Dictionary(docs)
#     corpus = [dictionary.doc2bow(doc) for doc in docs]
#     model_list, coherence_values = compute_coherence_values(
#         dictionary, corpus, docs, start=start, limit=limit, step=step
#     )
#     lsi = model_list[coherence_values.index(max(coherence_values))]
#     index = gensim.similarities.MatrixSimilarity(lsi[corpus])
#     similarity_matrix = np.zeros((len(corpus), len(corpus)))
#     for i, similarities in enumerate(index):
#         similarity_matrix[i] = similarities
#     return similarity_matrix


def calc_sim_mat(docs):
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lsi = gensim.models.LsiModel(corpus, num_topics=2, id2word=dictionary)
    index = gensim.similarities.MatrixSimilarity(
        lsi[corpus], num_features=len(dictionary)
    )
    sim_mat = np.zeros((len(corpus), len(corpus)))
    for i, similarities in enumerate(index):
        sim_mat[i] = similarities
    return sim_mat


def calc_acsm(sim_mat) -> float:
    return np.mean(sim_mat[np.triu_indices(len(sim_mat), k=1)])


def calc_c3(acsm) -> float:
    return max(acsm, 0)


def calc_lcsm(sim_mat: np.ndarray, acsm: float) -> int:
    n = sim_mat.shape[0]
    neighbors: list[set[int]] = []
    for i in range(n):
        indices = np.argwhere(sim_mat[i] > acsm).flatten()
        neighbors.append(set(j for j in indices if j != i))
    if all(len(n) == 0 for n in neighbors):
        return 0
    p, q = 0, 0
    for i, j in it.combinations(list(range(n)), 2):
        i_neighbors = neighbors[i]
        j_neighbors = neighbors[j]
        if len(i_neighbors & j_neighbors) == 0:
            p += 1
        else:
            q += 1
    return max(0, p - q)
