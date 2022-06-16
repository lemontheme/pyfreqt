from __future__ import annotations

import re
import struct
import itertools as it
import functools as ft
from dataclasses import dataclass, field, asdict as dataclass_as_dict
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Iterable, Optional, Iterator, TypedDict, Dict, Callable
import pickle

import msgpack
import lmdb
from tqdm import tqdm


@dataclass(slots=True)
class Node:
    value: Optional[str] = None
    sibling: int = -1
    child: int = -1
    parent: int = -1


@dataclass(slots=True)
class ProjectedTree:  # struct projected_t in reference implementation.
    depth: int = -1
    support: int = 0
    locations: List[Tuple[int, int]] = field(default_factory=list)
    n_nodes: int = -1


class _SubtreeRightMostChildLocation(TypedDict):
    txn_idx: int
    node_idx: int  # preorder


class SubtreeDict(TypedDict):
    pattern: str
    df: int
    tf: int  # 'weighted support' in reference inplementation.
    size: int  # i.e. number of nodes
    where: List[_SubtreeRightMostChildLocation]


def tokenize_s_expr(s_exp: str) -> Iterable[str]:
    """
    Example:
        >>> exp = "(A( B(C  )( D)) )"  # Notice inconsistent spacing.
        >>> tokenize_s_expr(exp)
        ['(', 'A', '(', 'B', '(', 'C', ')', '(', 'D', ')', ')', ')']
    """
    return [token for token in re.split(r"([()])", re.sub(r"\s", "", s_exp)) if token]


def drop_opening_pars(tokens: Iterable[str]) -> Iterable[str]:
    """Omit token corresponding to opening parentheses '('.

    These are not needed for the canonical code representation."""
    return [token for token in tokens if token != "("]


def parse_s_expr(s_expr: str) -> List[Node]:  # `str2node` in reference implementation.
    """Convert tree represented as symbolic expression to Nodes, where Nodes[0] = root.

    Example S-expression: ( A ( B ( D ) ( E ) ) ).

    """
    # 1. Tokenize
    #  - Omit opening parentheses.
    #  - Get count of symbols representing nodes.
    tokens = list(drop_opening_pars(tokenize_s_expr(s_expr)))
    n_nodes = sum(1 for t in tokens if t != ")")

    # 2. For each node token in the S-expression, add a `Node` instance to `nodes` and
    #    add a `-1` to `siblings`.
    #    Create list of ints `siblings`

    # Result
    nodes: List[Node] = [Node() for _ in range(n_nodes)]
    # Right-most child assoc'd with parent.
    siblings: List[int] = [-1] * n_nodes

    # 3. Main part. (Still figuring out what exactly is happening here.)
    stack = []  # 'sr' in reference implementation.
    node_idx = 0  # 'id' in reference.

    for token in tokens:
        if token == ")":
            if len(stack) < 2:
                continue

            parent_idx, child_idx = stack[-2:]
            nodes[child_idx].parent = parent_idx

            parent_node = nodes[parent_idx]
            if parent_node.child == -1:
                parent_node.child = child_idx

            prev_child_of_parent_idx = siblings[parent_idx]
            if prev_child_of_parent_idx != -1:
                nodes[prev_child_of_parent_idx].sibling = child_idx

            siblings[parent_idx] = child_idx
            stack = stack[:-1]  # drop child.

        else:  # In case of token representing node.
            nodes[node_idx].value = token
            stack.append(node_idx)
            node_idx += 1

    return nodes


@dataclass
class FREQTOriginal:
    min_support: int = 1
    min_nodes: int = 1  # `min_pat` in reference.
    max_nodes: int = 10e4  # `max_pat` in reference. doc: max pattern length
    weighted: bool = False  # Use weighted support.
    enc: bool = False  # Use internal string encoding format as output.
    prune_pred: Callable = None
    cache_dir: Optional[Path] = None
    _transaction_store: InMemoryTreeTransactionsStore | LMDBTreeTransactionsStore = field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self):
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            transactions_db_prefix = self.cache_dir / "transactions"
            self._transaction_store = LMDBTreeTransactionsStore(db_path=transactions_db_prefix)
        else:
            self._transaction_store = InMemoryTreeTransactionsStore()
        if self.prune_pred is None:
            self.prune_pred = lambda patt_toks, support, n_nodes: True

    # NEW
    def index_trees(self, s_exprs: Iterable[str], total: int = None) -> None:
        self._transaction_store.populate(
            parse_s_expr(s_expr) for s_expr in tqdm(s_exprs, total=total, mininterval=2.0)
        )

    # NEW
    def iter_subtrees(self) -> Iterator[SubtreeDict]:
        # Single-node tree occurence list
        freq1: defaultdict[tuple[str], ProjectedTree] = defaultdict(ProjectedTree)
        for i, transaction in self._transaction_store:
            for j, node in enumerate(transaction):
                freq1[(node.value,)].locations.append((i, j))

        # self._prune(freq1)

        for single_node_pattern, project_t in tqdm(freq1.items(), desc="Starting nodes explored"):
            project_t.depth = 0
            # project_t.n_nodes = 1
            pattern_toks: tuple[str] = single_node_pattern
            yield from self._project_iter_v2(project_t, pattern_toks)

    def _project_iter_v2(
        self, projected_tree: ProjectedTree, pattern_toks: tuple[str, ...]
    ) -> Iterator[SubtreeDict]:
        """Generate candidate."""
        stack: list[tuple[tuple[str, ...], ProjectedTree]] = [(pattern_toks, projected_tree)]
        candidates: defaultdict[tuple[str, ...], ProjectedTree] = defaultdict(ProjectedTree)

        min_nodes: int = self.min_nodes
        max_nodes: int = self.max_nodes

        while stack:
            pattern_toks, projected_tree = stack.pop()

            # <Pruning Step>
            support: int = self._compute_support(projected_tree)
            if support < self.min_support:
                continue
            n_nodes = sum(1 for tok in pattern_toks if tok != ")")
            if n_nodes > max_nodes:
                continue
            if not self.prune_pred(pattern_toks, support, n_nodes):
                continue
            # </Pruning Step>

            projected_tree.support = support
            projected_tree.n_nodes = n_nodes

            if min_nodes <= n_nodes:
                # In reference, this was side-effectual and was located in the body of
                # the for-loop over candidates. I've moved it here so that the calling
                # function iter_subtrees() doesn't have to duplicate filtering logic.
                yield self._report(projected_tree, pattern_toks)

            tree_depth = projected_tree.depth
            # Find all candidates by expanding right-most branch.
            # Convert candidate to internal string code.
            # INVESTIGATE: or just use tuples everywhere?

            # Re-use candidates dict, rather than re-allocating memory for new dict
            # in each iteration.
            # INVESTIGATE: Does this actually help?
            candidates.clear()

            for transaction_idx, initial_pos_idx in projected_tree.locations:
                pos_idx: int = initial_pos_idx
                prefix: tuple[str, ...] = ()
                current_depth: int = -1
                transaction_nodes: list[Node] = self._transaction_store[transaction_idx]
                while current_depth < tree_depth and pos_idx != -1:
                    start = (
                        transaction_nodes[pos_idx].child
                        if current_depth == -1
                        else transaction_nodes[pos_idx].sibling
                    )
                    new_depth = tree_depth - current_depth
                    next_node_idx = start  # 'l' in reference.
                    while next_node_idx != -1:
                        next_node: Node = transaction_nodes[next_node_idx]
                        item: tuple[str, ...] = prefix + (next_node.value,)
                        candidate = candidates[item]
                        candidate.locations.append((transaction_idx, next_node_idx))
                        candidate.depth = new_depth
                        # Finally
                        next_node_idx = next_node.sibling
                    if current_depth != -1:
                        pos_idx = transaction_nodes[pos_idx].parent  # Go up right-most branch.
                    prefix += (")",)
                    current_depth += 1

            for pattern, project_t in candidates.items():
                new_pattern_toks = pattern_toks + pattern
                stack.append((new_pattern_toks, project_t))

    # def _prune(self, candidates: dict[tuple[str, ...], ProjectedTree]) -> None:
    #     pattern: tuple[str, ...]
    #     candidate: ProjectedTree
    #     to_prune = []
    #     for pattern, project_t in candidates.items():
    #         support: int = self._compute_support(project_t)
    #         if support < self.min_support:
    #             to_prune.append(pattern)
    #             continue
    #         n_nodes = sum(1 for tok in pattern if tok != ")")
    #         if n_nodes > self.max_nodes:
    #             to_prune.append(pattern)
    #             continue
    #         # if not self.custom_criterion(patt_toks, support, n_nodes):
    #         #     to_prune.append(patt_toks)
    #         #     continue
    #         project_t.support = support
    #         project_t.n_nodes = n_nodes
    #     for pattern in to_prune:
    #         del candidates[pattern]

    def _compute_support(self, projected_tree: ProjectedTree) -> int:
        tree = projected_tree
        if self.weighted:
            return len(tree.locations)  # All occurences.
        old: Optional[int] = -1
        support: int = 0
        for transaction_idx, _ in tree.locations:
            if transaction_idx != old:  # Max one occurence per tree/transaction.
                support += 1
            old = transaction_idx
        return support

    @staticmethod
    def _report(
        projected_tree: ProjectedTree, pattern_toks: tuple[str]
    ) -> Optional[SubtreeDict]:
        s_exp: str = ""
        par_balance: int = 0
        for tok in pattern_toks:
            if tok == ")":
                par_balance -= 1
                s_exp = f"{s_exp})"
            else:
                s_exp = f"{s_exp}({tok}"
                par_balance += 1
        s_exp = f"{s_exp}{')' * max(0, par_balance)}"
        support = projected_tree.support
        weighted_support = len(projected_tree.locations)  # ?
        return SubtreeDict(
            pattern=s_exp,
            size=projected_tree.n_nodes,
            df=support,
            tf=weighted_support,
            where=[
                _SubtreeRightMostChildLocation(txn_idx=tree_i, node_idx=node_j)
                for tree_i, node_j in projected_tree.locations
            ],
        )


ArrayTree = list[Node]


# This does not improve __getitem__ performance. (In fact, it slows it down by 2x.)
def serialize_array_tree(array_tree: ArrayTree) -> bytes:
    return msgpack.packb([node.__dict__ for node in array_tree])


def deserialize_array_tree(data: bytes | memoryview) -> ArrayTree:
    return [Node(**x) for x in msgpack.unpackb(data)]
# #


class InMemoryTreeTransactionsStore:
    def __init__(self):
        self._transactions: list[ArrayTree] = []

    def populate(self, trees: Iterable[ArrayTree]) -> None:
        self._transactions.extend(trees)

    def __iter__(self) -> Iterator[tuple[int, ArrayTree]]:
        return zip(range(len(self._transactions)), self._transactions)

    def __getitem__(self, idx: int) -> ArrayTree:
        return self._transactions[idx]


class LMDBTreeTransactionsStore:

    _lmdb_env: lmdb.Environment

    def __init__(self, db_path: Path):
        db_path.mkdir(exist_ok=True, parents=True)
        for f in db_path.glob("*.mdb"):
            f.unlink()
        self._init_storage(db_path)

    def _init_storage(self, db_path):
        # fmt: off
        self._lmdb_env: lmdb.Environment = lmdb.open(
            path=str(db_path),
            map_size=int(2e10),  # 20 GB
            writemap=True,
            max_dbs=1,
            map_async=True
        )
        # fmt: on
        self._txn_db = self._lmdb_env.open_db(b"txns", integerkey=True, dupsort=False)

    def populate(self, trees: Iterable[ArrayTree]) -> None:
        with self._lmdb_env.begin(db=self._txn_db, write=True) as txn:
            cursor = txn.cursor()
            cursor.last()
            last_key_bin = cursor.value()
            if last_key_bin:
                (last_key,) = struct.unpack("n", last_key_bin)
                next_key = last_key + 1
            else:
                next_key = 0
            bin_keys_iter = (struct.pack("n", i) for i in it.count(next_key))
            bin_trees_iter = (pickle.dumps(t, protocol=5) for t in trees)
            # bin_trees_iter = (serialize_array_tree(t) for t in trees)
            index_stream = zip(bin_keys_iter, bin_trees_iter)
            cursor.putmulti(index_stream, dupdata=False, append=True)

    def __iter__(self) -> Iterator[tuple[int, ArrayTree]]:
        with self._lmdb_env.begin(db=self._txn_db, buffers=True) as txn:
            cursor = txn.cursor()
            cursor.first()
            for k_bin, v_bin in cursor:
                (k,) = struct.unpack("n", k_bin)
                nodes = pickle.loads(v_bin)
                # nodes = deserialize_array_tree(v_bin)
                yield k, nodes

    @ft.lru_cache(1_000)
    def __getitem__(self, idx: int) -> ArrayTree:
        k_bin = struct.pack("n", idx)
        with self._lmdb_env.begin(db=self._txn_db, buffers=True) as txn:
            v = txn.get(k_bin)
            if v is None:
                raise IndexError
            else:
                return pickle.loads(v)
                # return deserialize_array_tree(v)

    # @ft.cached_property
    # def __read_txn(self):
    #     return self._lmdb_env.begin(db=self._txn_db, buffers=True)


# class OnDiskList:
#
#     def append(self):
#         ...
#
#     def __iter__(self):
#         ...


#  -------------------------------- TESTS --------------------------------------------


# WIP
# def test_parse_s_expr():
#     s_expr = "( A ( B ( C ) ( D ) ) )"
#     nodes = parse_s_expr(s_expr)
#     print(s_expr, "->", nodes)
#     assert nodes[0].value == "A"
#     assert len(nodes) == 4
#     assert nodes[1].value == "B"
#     assert nodes[nodes[1].child].value == "C"
#
#
# # freqt = FREQTOriginal()
# #
# # db = [
# #     "( A ( B ( C ) ( D ) ) )",
# #     # "( A ( B ) ( D ) )"
# # ]
# #
# # freqt.run(db)
#
# def test_simple():
#     db = [
#         # "( A ( B ( C ( D ) ) ( E ) ) )",
#         # "( A ( B ( C ( D ) ) ) ( E ) )"
#
#         "( A ( B ( C ) ( E ) ) ( B ( D ) ( F ) ) )",
#         "( A ( B ( C ) ( F ) ) ( B ( D ) ( E ) ) )"
#
#         "( A ( B ( C ) ( E ) ) ( B ( D ) ( F ) ) )",
#         "( A ( B ( C ) ( F ) ) ( B ( D ) ( E ) ) )"
#
#
#         # "( A ( B ( C ( D ) ) ( B ( C ( D ) ) )",
#         # "( A ( B ) ( E ) )",
#         # "( B ( C ( D ) ) )",
#         # "( B ( C ( D ) ) )",
#     ]
#
#     freqt = FREQTOriginal(min_support=1, max_nodes=5, min_nodes=5, weighted=True)
#     freqt.index_trees(db)
#     for st in freqt.iter_subtrees():
#         print(st)
#
#
# test_simple()


# freqt = FREQT(**options)
# freqt.add_trees()  / .index_trees() / .register_trees()
# freqt.save()  # make pickleable
# cls FREQT.load() -> FREQT
# freqt.iter_subtrees() -> iter
# freqt.transactions
