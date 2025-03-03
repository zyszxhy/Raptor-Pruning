import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set, Optional, Tuple

import inspect
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import svd
from scipy.linalg import qr
import copy
import json
import umap

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

    # def remove_linearly_dependent_nodes(self, nodes, qr_threshold: float = 1e-5, dim: int = 5, num_neighbors: int = None, metric: str = "cosine"):
    #     embeddings = np.array([node.embeddings[self.cluster_embedding_model] for node in nodes])
    #     if num_neighbors is None:
    #         num_neighbors = max(int((len(embeddings) - 1) ** 0.5), 2)
    #     reduced_embeddings = umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)
    #     Q, R = qr(reduced_embeddings.T)
    #     remove_indices = np.where(np.abs(np.diag(R)) < qr_threshold)[0]
    
    #     return remove_indices
    
    def remove_linearly_dependent_nodes(self, nodes, threshold: float = 0.01, dim: int = 10, num_neighbors: int = None, metric: str = "cosine"):
        if len(nodes) <= 2:
            return []
        embeddings = np.array([node.embeddings[self.cluster_embedding_model] for node in nodes])
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
            lin_indep_values[i] = np.linalg.norm(r_i)
            q_array[:, i] = r_i / np.linalg.norm(r_i)
        
        remove_indices = np.where(np.array(lin_indep_values) < threshold)[0]
        return remove_indices
    
    def find_redundant_qr_nodes(self, nodes, threshold=0.8):
        node_index_to_remove = []
        active_nodes = [node for node in nodes if node.if_store and not node.children]

        if len(active_nodes) <= 1:
            return node_index_to_remove
    
        redundant_indices = self.remove_linearly_dependent_nodes(active_nodes, threshold)
        
        for index in redundant_indices:
            node_to_remove = active_nodes[index]
            for node in nodes:
                if node.index == node_to_remove.index:
                    node.if_store = False
                    node_index_to_remove.append(node.index)
                    # print(f"Pruned node with redundancy score {max_redundancy_score:.2f}")
        return node_index_to_remove
    
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
                nodes_score = []
                for node in cluster:
                    info_score = None
                    while(info_score is None):
                        # info_score = self.info_eval_model.eval_info_structure(summary=summarized_text, context=node.text)
                        info_score = self.info_eval_model.eval_info(summary=summarized_text, context=node.text)
                    # if info_score.score <= 3 and not node.children: # TODO change the threshold
                    # if int(info_score[0]) <= 2 and not node.children: # TODO change the threshold
                    #     node.if_store = False
                    #     node_index_to_remove.append(node.index)
                    if int(info_score[0]) <= 3 and 're_sum' in inspect.signature(Node.__init__).parameters:
                        re_sum = self.resum_model.Re_sum(summary=summarized_text, context=node.text)
                        node.re_sum = node.text
                        node.text = re_sum
                        node.embeddings = {model_name: model.create_embedding(node.text)
                                            for model_name, model in self.embedding_models.items()}

                    
                    if_familiar = None
                    while(if_familiar == None):
                        if_familiar = self.familiar_eval_model.eval_familiar(summary=summarized_text, context=node.text) # summary=summarized_text, 
                    
                    # if if_familiar.score <= 3 and node.index not in node_index_to_remove:  # TODO change the threshold of familiar
                    #     node.if_store = False
                    #     node_index_to_remove.append(node.index)
                    
                    nodes_score.append(int(if_familiar[0])) # int(info_score[0]) + 
                sorted_nodes = [node for node, score in sorted(zip(cluster, nodes_score), key=lambda x: x[1], reverse=True)]
                
                node_index_to_remove.extend(self.find_redundant_qr_nodes(sorted_nodes, threshold=0.005)) # TODO change the threshold of redundant
                # node_index_to_remove.extend(self.find_redundant_nodes(cluster, threshold=0.80)) # TODO change the threshold of redundant


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

            # resum count
            if hasattr(all_nodes[0], 're_sum'):
                ori_text_len = 0
                resum_text_len = 0
                ori_text_token = 0
                resum_text_token = 0
            # hypo_qs count
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
                if hasattr(all_nodes[0], 're_sum') and node.re_sum:
                    ori_text_len += len(node.re_sum)
                    resum_text_len += text_len
                    ori_text_token += len(self.tokenizer.encode(node.re_sum))
                    resum_text_token += text_token
                    # node.text = node.re_sum

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
            if hasattr(all_nodes[0], 're_sum'):
                print(f"Resumd {resum_text_len}/{ori_text_len} text length, reduced to {resum_text_len/ori_text_len * 100:.2f}%.")
                print(f"Resumd {resum_text_token}/{ori_text_token} text token, reduced to {resum_text_token/ori_text_token * 100:.2f}%.")
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
            if hasattr(all_nodes[0], 're_sum'):
                result.update({
                    "ori_text_len": ori_text_len,
                    "resum_text_len": resum_text_len,
                    "ori_text_token": ori_text_token,
                    "resum_text_token": resum_text_token,
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
            # for node in layer_nodes:
            #     if node.re_sum:
            #         node.text = node.re_sum
            all_nodes_removed[layer] = layer_nodes
        
        return all_nodes_removed
    
    def remove_redundant_nodes_dict(self, all_nodes: Dict):
        remove_list = []
        for index, node in all_nodes.items():
            if not node.if_store:
                remove_list.append(index)
            # if node.re_sum:
            #     node.text = node.re_sum
        for index in remove_list:
            del all_nodes[index]
        
        return all_nodes

