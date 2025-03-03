import logging
import os
from typing import Dict, List, Set

import numpy as np
import umap

import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings, get_hypo_qs_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances,
                    reverse_mapping)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeRetrieverConfig:
    def __init__(
        self,
        tokenizer=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        context_embedding_model=None,
        embedding_model=None,
        num_layers=None,
        start_layer=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "selection_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.selection_mode = selection_mode

        if context_embedding_model is None:
            context_embedding_model = "OpenAI"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        self.context_embedding_model = context_embedding_model

        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError("start_layer must be an integer and at least 0")
        self.start_layer = start_layer

    def log_config(self):
        config_log = """
        TreeRetrieverConfig:
            Tokenizer: {tokenizer}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Context Embedding Model: {context_embedding_model}
            Embedding Model: {embedding_model}
            Num Layers: {num_layers}
            Start Layer: {start_layer}
        """.format(
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            context_embedding_model=self.context_embedding_model,
            embedding_model=self.embedding_model,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
        )
        return config_log


class TreeRetriever(BaseRetriever):

    def __init__(self, config, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )

        self.tree = tree
        self.num_layers = (
            config.num_layers if config.num_layers is not None else tree.num_layers + 1
        )
        self.start_layer = (
            config.start_layer if config.start_layer is not None else tree.num_layers
        )

        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)

        logging.info(
            f"Successfully initialized TreeRetriever with Config {config.log_config()}"
        )

    def create_embedding(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.embedding_model.create_embedding(text)

    def remove_linearly_dependent_nodes(self, nodes, dim: int = 10, num_neighbors: int = None, metric: str = "cosine"):
        if len(nodes) <= 2:
            return []
        embeddings = np.array([node.embeddings[self.context_embedding_model] for node in nodes])
        if num_neighbors is None:
            num_neighbors = max(int((len(embeddings) - 1) ** 0.5), 2)
        if len(embeddings) <= dim + 1:
            reduced_embeddings = umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric, init='random').fit_transform(embeddings).T
        else:
            reduced_embeddings = umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric).fit_transform(embeddings).T
        # reduced_embeddings = embeddings.T
        m, n = reduced_embeddings.shape
        lin_indep_values = np.zeros(n)
        q_array = np.zeros((m, n))
        q_array[:, 0] = reduced_embeddings[:, 0]
        lin_indep_values[0] = 1.0

        for i in range(1, n):
            a_i = reduced_embeddings[:, i]
            projection = np.zeros(m)
            
            for j in range(i):
                q_j = q_array[:, j]
                if np.all(q_j==0):
                    continue
                # q_j_normalized = q_j / np.linalg.norm(q_j)
                # proj_component = np.dot(a_i, q_j_normalized) * q_j_normalized
                proj_component = (np.dot(a_i, q_j) / np.dot(q_j, q_j)) * q_j
                projection += proj_component
            r_i = a_i - projection
            lin_indep_values[i] = np.linalg.norm(r_i) / np.linalg.norm(a_i)
            q_array[:, i] = r_i
        
        return lin_indep_values
    
    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            query (str): The query text.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)

        if hasattr(node_list[0], 'hypo_qs'):
            embeddings = get_hypo_qs_embeddings(node_list, self.context_embedding_model)
        else:
            embeddings = get_embeddings(node_list, self.context_embedding_model)

        distances = distances_from_embeddings(query_embedding, embeddings)

        indices = indices_of_nearest_neighbors_from_distances(distances)

        selected_nodes_1 = []
        for idx in indices[:top_k * 2]:
            node = node_list[idx]
            selected_nodes_1.append(node)
        lin_indep_values = self.remove_linearly_dependent_nodes(selected_nodes_1)
        lin_index = np.argsort(-lin_indep_values)

        total_tokens = 0
        # for idx in indices[:top_k]:
        for idx in lin_index[:top_k]:

            # node = node_list[idx]
            node = selected_nodes_1[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: List[Node], query: str, num_layers: int
    ) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            current_nodes (List[Node]): A List of the current nodes.
            query (str): The query text.
            num_layers (int): The number of layers to traverse.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = current_nodes

        for layer in range(num_layers):

            if hasattr(node_list[0], 'hypo_qs'):
                embeddings = get_hypo_qs_embeddings(node_list, self.context_embedding_model)
            else:
                embeddings = get_embeddings(node_list, self.context_embedding_model)

            distances = distances_from_embeddings(query_embedding, embeddings)

            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.selection_mode == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.threshold
                ]

            elif self.selection_mode == "top_k":
                best_indices = indices[: self.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]

            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:

                child_nodes = []

                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                # take the unique values
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10, 
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> str:
        """
        Queries the tree and returns the most relevant information.

        Args:
            query (str): The query text.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            collapse_tree (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The result of the query.
        """

        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        # Set defaults
        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.tree.num_layers
        ):
            raise ValueError(
                "start_layer must be an integer between 0 and tree.num_layers"
            )

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        if collapse_tree:
            logging.info(f"Using collapsed_tree")
            selected_nodes, context = self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = self.retrieve_information(
                layer_nodes, query, num_layers
            )

        if return_layer_information:

            layer_information = []

            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[node.index],
                    }
                )

            return context, layer_information

        return context
