from typing import Dict, List, Set


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(self, text: str, index: int, children: Set[int], embeddings, if_store: bool = True, re_sum: str = None) -> None: # hypo_qs: str, hypo_qs_embeddings, 
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings
        # self.hypo_qs = hypo_qs
        # self.hypo_qs_embeddings = hypo_qs_embeddings
        self.if_store = if_store
        self.re_sum = re_sum


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes
