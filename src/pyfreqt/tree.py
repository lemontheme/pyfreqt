from __future__ import annotations

import re
import operator as op
from dataclasses import dataclass, field
from typing import Sequence, TypeVar, Callable, Optional, Iterable, Any

T = TypeVar("T")
V = TypeVar("V")


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


# def tree_to_s_expr_old(
#     root: T,
#     val_getter: Callable[[T], T] = op.attrgetter("val"),
#     children_getter: Callable[[T], Sequence[T]] = op.attrgetter("children"),
# ) -> str:
#     nodes = [(None, root)]  # Parent traversal idx, node
#     traversal = []  # parent traversal idx, node val, children exps
#     while nodes:
#         parent_idx, node = nodes.pop()
#         traversal.append((parent_idx, node, []))
#         node_idx = len(traversal) - 1
#         for child in reversed(children_getter(node)):
#             nodes.append((node_idx, child))
#     result: str
#     while traversal:
#         parent_idx, node, children = traversal.pop()
#         node_val = val_getter(node)
#         if children:
#             result = f"( {node_val} {' '.join(c for c in reversed(children))} )"  # Double use of `reversed()`
#         else:
#             result = f"( {node_val} )"
#         if parent_idx is None:
#             return result
#         else:
#             traversal[parent_idx][2].append(result)

class SExpression:

    @classmethod
    def from_tree(cls, tree) -> SExpression:
        ...

    def to_string(self):
        ...

    def to_lcrs_tree(self):
        ...

    def to_n_ary_tree(self):
        ...


# O(d) memory usage where d = depth of deepest branch
def tree_to_s_expr(
    node: T,
    val_getter: Callable[[T], V] = op.attrgetter("val"),
    children_getter: Callable[[T], Sequence[T]] = op.attrgetter("children")
) -> str:
    next_nodes: list[Optional[T]] = [node]
    result: str = ""
    while next_nodes:
        # See commend about significance of `None` below.
        current_node: Optional[T] = next_nodes.pop()
        if current_node is None:
            result = f"{result}) "
        else:
            node_val: V = val_getter(current_node)
            children: Sequence[T] = children_getter(current_node)
            if children:
                result = f"{result}( {node_val} "
                next_nodes.append(None)  # `None` signals the end of current depth.
                next_nodes.extend(reversed(children))
            else:
                result = f"{result}( {node_val} ) "
    return result.strip()


@dataclass
class Node:
    val: str = None
    children: list[Node] = field(default_factory=list)
    parent: Node = None


example_tree = Node("A", [Node("B", [Node("C", [])]), Node("D", [])])

tree_to_s_expr(example_tree)

# n_ary_tree vs. right_sib_tree
# Simple tree
# left-child, right-sibling representation (LCRS)

# def s_expr_to_lcrs_tree(): ...


def s_expr_to_n_ary_tree(
    s_expr: str,
    node_factory: Callable[[], T] = Node,
    val_setter: Callable[[T, V], None] = lambda node, val: setattr(node, "val", val),
    child_adder: Callable[[T, T], None] = lambda node, child: node.children.append(child),
    parent_setter: Optional[Callable[[T, T], None]] = lambda node, parent: setattr(node, "parent", parent)
) -> T:
    tokens: Iterable[str] = drop_opening_pars(tokenize_s_expr(s_expr))
    parent_setter = parent_setter or (lambda n, p: None)
    nodes: list[T] = []
    stack: list[T] = []
    for token in tokens:
        if token == ")" and len(stack) >= 2:
            parent, child = stack[-2:]
            parent_setter(child, parent)
            child_adder(parent, child)
            stack = stack[:-1]
        else:
            node = node_factory()
            val_setter(node, token)
            # nodes.append(node)
            stack.append(node)
    return stack[0]  # root node


def s_expr_to_lcrs_tree():
    ...


def s_expr_to_freqt_lcrs_nodes(s_expr):
    # 1. Tokenize
    #  - Omit opening parentheses.
    #  - Get count of symbols representing nodes.
    tokens = list(drop_opening_pars(tokenize_s_expr(s_expr)))
    n_nodes = sum(1 for t in tokens if t != ")")

    # 2. For each node token in the S-expression, add a `Node` instance to `nodes` and
    #    add a `-1` to `siblings`.
    #    Create list of ints `siblings`
    # Result
    nodes: list[Node] = [Node() for _ in range(n_nodes)]
    # Right-most child assoc'd with parent.
    siblings: list[int] = [-1] * n_nodes

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
            nodes[node_idx].val = token
            stack.append(node_idx)
            node_idx += 1

    return nodes

# def parse_s_expr(s_expr: str) -> list[Node]:  # `str2node` in reference implementation.
#     """Convert tree represented as symbolic expression to Nodes, where Nodes[0] = root.
#
#     Example S-expression: ( A ( B ( D ) ( E ) ) ).
#
#     """
#     # 1. Tokenize
#     #  - Omit opening parentheses.
#     #  - Get count of symbols representing nodes.
#     tokens = list(drop_opening_pars(tokenize_s_expr(s_expr)))
#     n_nodes = sum(1 for t in tokens if t != ")")
#
#     # 2. For each node token in the S-expression, add a `Node` instance to `nodes` and
#     #    add a `-1` to `siblings`.
#     #    Create list of ints `siblings`
#
#     # Result
#     nodes: List[Node] = [Node() for _ in range(n_nodes)]
#     # Right-most child assoc'd with parent.
#     siblings: List[int] = [-1] * n_nodes
#
#     # 3. Main part. (Still figuring out what exactly is happening here.)
#     stack = []  # 'sr' in reference implementation.
#     node_idx = 0  # 'id' in reference.
#
#     for token in tokens:
#         if token == ")":
#             if len(stack) < 2:
#                 continue
#
#             parent_idx, child_idx = stack[-2:]
#             nodes[child_idx].parent = parent_idx
#
#             parent_node = nodes[parent_idx]
#             if parent_node.child == -1:
#                 parent_node.child = child_idx
#
#             prev_child_of_parent_idx = siblings[parent_idx]
#             if prev_child_of_parent_idx != -1:
#                 nodes[prev_child_of_parent_idx].sibling = child_idx
#
#             siblings[parent_idx] = child_idx
#             stack = stack[:-1]  # drop child.
#
#         else:  # In case of token representing node.
#             nodes[node_idx].value = token
#             stack.append(node_idx)
#             node_idx += 1
#
#     return nodes
