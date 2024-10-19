import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set, Optional, Tuple

import inspect
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy
import json

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def calculate_redundancy_scores(self, nodes):
        embeddings = [node.embeddings[self.cluster_embedding_model] for node in nodes]
        similarity_matrix = cosine_similarity(embeddings)
        redundancy_scores = np.sum(similarity_matrix, axis=1) / similarity_matrix.shape[1]
        return redundancy_scores 
    
    def find_redundant_nodes(self, nodes, threshold=0.8):
        node_index_to_remove = []
        while True:
            active_nodes = [node for node in nodes if node.if_store]

            if len(active_nodes) <= 1:
                break
        
            redundancy_scores = self.calculate_redundancy_scores(active_nodes)
            max_redundancy_index = np.argmax(redundancy_scores)
            max_redundancy_score = redundancy_scores[max_redundancy_index]
            
            if max_redundancy_score > threshold:
                node_to_remove = active_nodes[max_redundancy_index]
                for node in nodes:
                    if node.index == node_to_remove.index:
                        node.if_store = False
                        node_index_to_remove.append(node.index)
                        # print(f"Pruned node with redundancy score {max_redundancy_score:.2f}")
            else:
                break
        return node_index_to_remove
    
    def create_parent_node(
        self, index: int, text: str, all_text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        
        if children_indices is None:
            children_indices = set()

        embeddings = {
            model_name: model.create_embedding(text)
            for model_name, model in self.embedding_models.items()
        }
        if 'hypo_qs' in inspect.signature(Node.__init__).parameters:
            hypo_qs = self.generate_hypo_qs(all_text)
            hypo_qs_embeddings = {
                model_name: model.create_embedding(hypo_qs)
                for model_name, model in self.embedding_models.items()
            }
            return (index, Node(text, index, children_indices, embeddings, hypo_qs, hypo_qs_embeddings))
        return (index, Node(text, index, children_indices, embeddings))
    
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        leaf_nodes: Dict[int, Node],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)
        node_index_to_remove = []

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock, node_index_to_remove: list
        ):
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            if 'if_store' in inspect.signature(Node.__init__).parameters:
                for node in cluster:
                    # info_score = None
                    # while(info_score is None):
                    #     info_score = self.info_eval_model.eval_info(summary=summarized_text, context=node.text)
                    # if info_score.score <= 3 and not node.children: # TODO change the threshold
                    #     node.if_store = False
                    #     node_index_to_remove.append(node.index)
                    
                    if_familiar = None
                    while(if_familiar == None):
                        if_familiar = self.familiar_eval_model.eval_familiar(summary=summarized_text, context=node.text) # summary=summarized_text, 
                    if if_familiar.score <= 1:  # TODO change the threshold of familiar
                        node.if_store = False
                        node_index_to_remove.append(node.index)
                
                # node_index_to_remove.extend(self.find_redundant_nodes(cluster, threshold=0.85)) # TODO change the threshold of redundant


                __, new_parent_node = self.create_parent_node(
                    next_node_index, summarized_text, summarized_text, {node.index for node in cluster if node.if_store}
                )
            else:
                __, new_parent_node = self.create_parent_node(
                    next_node_index, summarized_text, summarized_text, {node.index for node in cluster}
                )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                            node_index_to_remove,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                        node_index_to_remove,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            layer_to_nodes[layer] = list(current_level_nodes.values())
            if layer == 0:
                leaf_nodes = copy.deepcopy(current_level_nodes)
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes, leaf_nodes, node_index_to_remove

    def remove_redundant_nodes(self, all_nodes: Dict, node_index_to_remove: List[int], out_json: str = './count_reduce.json'):
        with open(out_json, 'a', encoding='utf-8') as output_f:
            total_text_len = 0
            total_text_token = 0
            remove_text_len = 0
            remove_text_token = 0
            if hasattr(all_nodes[0], 'hypo_qs'):
                total_hypo_qs_len = 0
                total_hypo_qs_token = 0
                remove_hypo_qs_len = 0
                remove_hypo_qs_token = 0
            for index, node in all_nodes.items():
                text_len = len(node.text)
                total_text_len += text_len
                text_token = len(self.tokenizer.encode(node.text))
                total_text_token += text_token
                if hasattr(all_nodes[0], 'hypo_qs'):
                    hypo_qs_len = len(node.hypo_qs)
                    total_hypo_qs_len += hypo_qs_len
                    hypo_qs_token = len(self.tokenizer.encode(node.hypo_qs))
                    total_hypo_qs_token += hypo_qs_token

                if int(index) in node_index_to_remove:
                    remove_text_len += text_len
                    remove_text_token += text_token
                    if hasattr(all_nodes[0], 'hypo_qs'):
                        remove_hypo_qs_len += hypo_qs_len
                        remove_hypo_qs_token += hypo_qs_token
            
            print(f"Removed {remove_text_len}/{total_text_len} text length, reduced {remove_text_len/total_text_len * 100:.2f}%.")
            print(f"Removed {remove_text_token}/{total_text_token} text token, reduced {remove_text_token/total_text_token * 100:.2f}%.")
            if hasattr(all_nodes[0], 'hypo_qs'):
                print(f"Removed {remove_hypo_qs_len}/{total_hypo_qs_len} hypo question length, reduced {remove_hypo_qs_len/total_hypo_qs_len * 100:.2f}%.")
                print(f"Removed {remove_hypo_qs_token}/{total_hypo_qs_token} hypo question token, reduced {remove_hypo_qs_token/total_hypo_qs_token * 100:.2f}%.")
            print(f"Removed {len(node_index_to_remove)}/{len(all_nodes)} nodes, reduced {len(node_index_to_remove)/len(all_nodes) * 100:.2f}%.")

            result = {
                "total_text_len": total_text_len,
                "total_text_token": total_text_token, 
                "remove_text_len": remove_text_len,
                "remove_text_token": remove_text_token,
                "total_node": len(all_nodes),
                "remove_node": len(node_index_to_remove)
            }
            if hasattr(all_nodes[0], 'hypo_qs'):
                result.update({
                    "total_hypo_qs_len": total_hypo_qs_len,
                    "total_hypo_qs_token": total_hypo_qs_token,
                    "remove_hypo_qs_len":remove_hypo_qs_len,
                    "remove_hypo_qs_token": remove_hypo_qs_token,
                })
            output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_f.flush()
        
        for remove_index in node_index_to_remove:
            del all_nodes[remove_index]
        
        return all_nodes
    
    def remove_redundant_nodes_list(self, all_nodes: Dict):
        all_nodes_removed = {}
        for layer, layer_nodes in all_nodes.items():
            layer_nodes = [node for node in layer_nodes if node.if_store]
            all_nodes_removed[layer] = layer_nodes
        
        return all_nodes_removed
    
    def remove_redundant_nodes_dict(self, all_nodes: Dict):
        remove_list = []
        for index, node in all_nodes.items():
            if not node.if_store:
                remove_list.append(index)
        for index in remove_list:
            del all_nodes[index]
        
        return all_nodes

