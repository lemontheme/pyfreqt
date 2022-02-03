import re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Tuple, List, Iterable, Optional, Iterator, TypedDict


@dataclass(slots=True)
class Node:
    value: Optional[str] = None
    sibling: int = -1
    child: int = -1
    parent: int = -1


@dataclass(slots=True)
class ProjectedTree:  # struct projected_t in reference implementation.
    # pattern_tokens: Optional[tuple[str]] = None
    # n_nodes: int = 0
    depth: int = -1
    support: int = 0
    locations: List[Tuple[int, int]] = field(default_factory=list)


class _SubtreeRootLocation(TypedDict):
    transaction_idx: int
    pre_order_idx: int


class SubtreeDict(TypedDict):
    pattern: str
    support_df: int
    support_tf: int  # 'weighted support' in reference inplementation.
    size: int  # i.e. number of nodes
    where: List[_SubtreeRootLocation]


def tokenize_s_expr(s_exp: str) -> Iterable[str]:
    """
    Example:
        >>> exp = "(A( B(C  )( D)) )"  # Notice inconsistent spacing.
        >>> tokenize_s_expr(exp)
        ['(', 'A', '(', 'B', '(', 'C', ')', '(', 'D', ')', ')', ')']
    """
    return [
        token for token in
        re.split(
            r"([()])",
            re.sub(r"\s", "", s_exp)
        )
        if token
    ]


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
    _transactions: List[List[Node]] = field(init=False, repr=False, default_factory=list)
    # _pattern: List[str] = field(init=False, repr=False)  # current pattern tokens
    # Config
    min_support: int = 1
    min_nodes: int = 1  # `min_pat` in reference.
    max_nodes: int = 10e4  # `max_pat` in reference. doc: max pattern length
    weighted: bool = False  # Use weighted support.
    enc: bool = False  # Use internal string encoding format as output.

    # NEW
    def index_trees(self, s_exprs: Iterable[str]) -> None:
        self._transactions.extend(
            parse_s_expr(s_expr) for s_expr in s_exprs
        )

    # NEW
    def iter_subtrees(self) -> Iterator[SubtreeDict]:
        # Single-node tree occurence list
        freq1: defaultdict[str, ProjectedTree] = defaultdict(ProjectedTree)
        for i, transaction in enumerate(self._transactions):
            for j, node in enumerate(transaction):
                freq1[node.value].locations.append(
                    (i, j)
                )

        self._prune(freq1)

        for single_node_pattern, project_t in freq1.items():
            project_t.depth = 0
            pattern_toks: tuple[str] = (single_node_pattern,)
            yield from self._project_iter_v2(project_t, pattern_toks)

    def _prune(self, candidates: dict[str, ProjectedTree]) -> None:
        pattern: str
        candidate: ProjectedTree
        to_prune = []
        for item, project_t in candidates.items():
            support: int = self._compute_support(project_t)
            if support < self.min_support:
                to_prune.append(item)
            else:
                project_t.support = support
        for item in to_prune:
            del candidates[item]

    def _project_iter_v2(self, projected_tree: ProjectedTree, pattern_toks: tuple[str]) -> None:
        """Generate candidate."""
        stack: list[tuple[tuple[str], ProjectedTree]] = [(pattern_toks, projected_tree)]
        candidates: defaultdict[str, ProjectedTree] = defaultdict(ProjectedTree)

        min_nodes: int = self.min_nodes
        max_nodes: int = self.max_nodes

        while stack:
            pattern_toks, projected_tree = stack.pop()
            # No `else` to limit indentation.
            # action != "resize" -> action == "project"
            n_nodes = len(pattern_toks)  # sum(1 for tok in pattern_toks if tok != ")")

            if max_nodes >= n_nodes >= min_nodes:
                # In reference, this was side-effectual and was located in the body of
                # the for-loop over candidates. I've moved it here so that the calling
                # function iter_subtrees() doesn't have to duplicate filtering logic.
                yield self._report(projected_tree, pattern_toks, n_nodes)
            if n_nodes >= max_nodes:
                continue

            tree_depth = projected_tree.depth
            # Find all candidates by expanding right-most branch.
            # Convert candidate to internal string code.

            # Re-use candidates dict, rather than re-allocating memory for new dict
            # in each iteration.
            candidates.clear()

            for transaction_idx, initial_pos_idx in projected_tree.locations:
                pos_idx: int = initial_pos_idx
                prefix: str = ""
                current_depth: int = -1
                transaction_nodes: list[Node] = self._transactions[transaction_idx]
                while current_depth < tree_depth and pos_idx != -1:
                    start = (
                        transaction_nodes[pos_idx].child
                        if current_depth == -1 else transaction_nodes[pos_idx].sibling
                    )
                    new_depth = tree_depth - current_depth
                    next_node_idx = start  # 'l' in reference.
                    while next_node_idx != -1:
                        item = f"{prefix} {transaction_nodes[next_node_idx].value}"
                        candidate = candidates[item]
                        candidate.locations.append((transaction_idx, next_node_idx))
                        candidate.depth = new_depth
                        # Finally
                        next_node_idx = transaction_nodes[next_node_idx].sibling
                    if current_depth != -1:
                        node = transaction_nodes[pos_idx]
                        pos_idx = node.parent
                    prefix = f"{prefix} )"
                    current_depth += 1

            self._prune(candidates)

            pattern: str
            project_t: ProjectedTree

            for pattern, project_t in candidates.items():
                new_pattern_toks = pattern_toks + tuple(pattern.split())
                stack.append(
                    (new_pattern_toks, project_t)
                )

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
        projected_tree: ProjectedTree, pattern_toks: tuple[str], n_nodes: int
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
            support_df=support,
            size=n_nodes,
            support_tf=weighted_support,
            where=[
                _SubtreeRootLocation(transaction_idx=tree_i, pre_order_idx=node_j)
                for tree_i, node_j in projected_tree.locations
            ]
        )


def main():
    ...

#  -------------------------------- TESTS --------------------------------------------


# WIP
def test_parse_s_expr():
    s_expr = "( A ( B ( C ) ( D ) ) )"
    nodes = parse_s_expr(s_expr)
    print(s_expr, "->", nodes)
    assert nodes[0].value == "A"
    assert len(nodes) == 4
    assert nodes[1].value == "B"
    assert nodes[nodes[1].child].value == "C"


# freqt = FREQTOriginal()
#
# db = [
#     "( A ( B ( C ) ( D ) ) )",
#     # "( A ( B ) ( D ) )"
# ]
#
# freqt.run(db)

def test_simple():
    db = [
        "( A ( B ( C ( D ) ) ( E ) ) )",
        "( A ( B ( C ( D ) ) ( B ( C ( D ) ) )",
        "( A ( B ) ( E ) )",
        "( B ( C ( D ) ) )",
        "( B ( C ( D ) ) )",
    ]

    freqt = FREQTOriginal(min_support=1, max_nodes=4, min_nodes=2, weighted=True)
    freqt.index_trees(db)
    for st in freqt.iter_subtrees():
        print(st)


test_simple()


# freqt = FREQT(**options)
# freqt.add_trees()  / .index_trees() / .register_trees()
# freqt.save()  # make pickleable
# cls FREQT.load() -> FREQT
# freqt.iter_subtrees() -> iter
# freqt.transactions
