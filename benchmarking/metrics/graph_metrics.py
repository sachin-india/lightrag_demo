"""
Graph-specific metrics for LightRAG evaluation

Metrics designed specifically for graph-based RAG systems.
Measures entity coverage, relationship quality, graph connectivity, and graph-based retrieval effectiveness.

NEW module created for LightRAG benchmarking system.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Set, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import json


class GraphMetrics:
    """Metrics specific to graph-based RAG systems like LightRAG"""
    
    def __init__(self):
        self.graph_data = None
        self.entity_types = set()
        self.relation_types = set()
    
    def load_graph_from_graphml(self, graphml_path: str) -> Dict[str, Any]:
        """
        Load and parse GraphML file
        
        Args:
            graphml_path: Path to GraphML file
            
        Returns:
            Dictionary with graph structure
        """
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        
        # Parse namespace
        ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        nodes = {}
        edges = []
        
        # Parse nodes
        for node in root.findall('.//graphml:node', ns):
            node_id = node.get('id')
            node_data = {'id': node_id}
            
            # Extract node attributes
            for data in node.findall('graphml:data', ns):
                key = data.get('key')
                value = data.text
                node_data[key] = value
            
            nodes[node_id] = node_data
        
        # Parse edges
        for edge in root.findall('.//graphml:edge', ns):
            source = edge.get('source')
            target = edge.get('target')
            edge_data = {'source': source, 'target': target}
            
            # Extract edge attributes
            for data in edge.findall('graphml:data', ns):
                key = data.get('key')
                value = data.text
                edge_data[key] = value
            
            edges.append(edge_data)
        
        self.graph_data = {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
        
        return self.graph_data
    
    def calculate_graph_statistics(self, graph_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate basic graph statistics
        
        Args:
            graph_data: Optional graph data (uses loaded graph if None)
            
        Returns:
            Dictionary with graph statistics
        """
        if graph_data is None:
            graph_data = self.graph_data
        
        if graph_data is None:
            raise ValueError("No graph data loaded. Call load_graph_from_graphml first.")
        
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        
        # Build adjacency information
        out_degree = defaultdict(int)
        in_degree = defaultdict(int)
        
        for edge in edges:
            out_degree[edge['source']] += 1
            in_degree[edge['target']] += 1
        
        # Calculate degrees
        degrees = []
        for node_id in nodes:
            degree = out_degree[node_id] + in_degree[node_id]
            degrees.append(degree)
        
        # Calculate statistics
        num_nodes = len(nodes)
        num_edges = len(edges)
        avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
        
        # Density: actual edges / possible edges
        max_edges = num_nodes * (num_nodes - 1)  # directed graph
        density = num_edges / max_edges if max_edges > 0 else 0
        
        # Connected components (simplified - just check for isolated nodes)
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge['source'])
            connected_nodes.add(edge['target'])
        
        isolated_nodes = num_nodes - len(connected_nodes)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'density': density,
            'isolated_nodes': isolated_nodes,
            'connectivity_ratio': len(connected_nodes) / num_nodes if num_nodes > 0 else 0
        }
    
    def entity_coverage_score(self, 
                             retrieved_entities: List[str], 
                             ground_truth_entities: Set[str]) -> Dict[str, float]:
        """
        Calculate entity coverage metrics
        
        Args:
            retrieved_entities: Entities retrieved by the system
            ground_truth_entities: Expected relevant entities
            
        Returns:
            Dictionary with coverage metrics
        """
        if not ground_truth_entities:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'coverage': 0.0}
        
        retrieved_set = set(retrieved_entities)
        overlap = retrieved_set.intersection(ground_truth_entities)
        
        precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(overlap) / len(ground_truth_entities)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        coverage = len(overlap) / len(ground_truth_entities)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'coverage': coverage,
            'num_retrieved': len(retrieved_set),
            'num_relevant': len(overlap),
            'num_ground_truth': len(ground_truth_entities)
        }
    
    def relation_coverage_score(self, 
                               retrieved_relations: List[Tuple[str, str, str]], 
                               ground_truth_relations: Set[Tuple[str, str, str]]) -> Dict[str, float]:
        """
        Calculate relation coverage metrics
        
        Args:
            retrieved_relations: Relations retrieved (source, relation_type, target)
            ground_truth_relations: Expected relevant relations
            
        Returns:
            Dictionary with relation coverage metrics
        """
        if not ground_truth_relations:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'coverage': 0.0}
        
        retrieved_set = set(retrieved_relations)
        overlap = retrieved_set.intersection(ground_truth_relations)
        
        precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(overlap) / len(ground_truth_relations)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        coverage = len(overlap) / len(ground_truth_relations)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'coverage': coverage,
            'num_retrieved': len(retrieved_set),
            'num_relevant': len(overlap),
            'num_ground_truth': len(ground_truth_relations)
        }
    
    def graph_traversal_quality(self, 
                                query: str,
                                retrieved_subgraph: Dict[str, Any],
                                relevance_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate quality of retrieved subgraph
        
        Args:
            query: Original query
            retrieved_subgraph: Subgraph returned by system
            relevance_threshold: Threshold for considering nodes relevant
            
        Returns:
            Subgraph quality metrics
        """
        nodes = retrieved_subgraph.get('nodes', [])
        edges = retrieved_subgraph.get('edges', [])
        
        if not nodes:
            return {'subgraph_density': 0.0, 'connectivity': 0.0, 'coherence': 0.0}
        
        # Calculate subgraph density
        num_nodes = len(nodes)
        num_edges = len(edges)
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0.0
        
        # Calculate connectivity (simplified)
        connected_components = self._count_connected_components(nodes, edges)
        connectivity = 1.0 - (connected_components - 1) / num_nodes if num_nodes > 1 else 1.0
        
        # Coherence: ratio of nodes with at least one edge
        nodes_with_edges = set()
        for edge in edges:
            nodes_with_edges.add(edge['source'])
            nodes_with_edges.add(edge['target'])
        
        coherence = len(nodes_with_edges) / num_nodes if num_nodes > 0 else 0.0
        
        return {
            'subgraph_density': density,
            'connectivity': connectivity,
            'coherence': coherence,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'connected_components': connected_components
        }
    
    def _count_connected_components(self, nodes: List[Dict], edges: List[Dict]) -> int:
        """Count number of connected components in subgraph"""
        if not nodes:
            return 0
        
        # Build adjacency list
        adj = defaultdict(set)
        node_ids = set()
        
        for node in nodes:
            node_id = node.get('id', str(node))
            node_ids.add(node_id)
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in node_ids and target in node_ids:
                adj[source].add(target)
                adj[target].add(source)  # Treat as undirected
        
        # DFS to count components
        visited = set()
        components = 0
        
        def dfs(node):
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
        
        for node_id in node_ids:
            if node_id not in visited:
                dfs(node_id)
                components += 1
        
        return components
    
    def entity_type_distribution(self, entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate distribution of entity types
        
        Args:
            entities: List of entity dictionaries with 'type' field
            
        Returns:
            Dictionary mapping entity types to counts
        """
        type_counts = Counter()
        
        for entity in entities:
            entity_type = entity.get('type', entity.get('entity_type', 'UNKNOWN'))
            type_counts[entity_type] += 1
        
        return dict(type_counts)
    
    def relation_type_distribution(self, relations: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate distribution of relation types
        
        Args:
            relations: List of relation dictionaries with 'type' field
            
        Returns:
            Dictionary mapping relation types to counts
        """
        type_counts = Counter()
        
        for relation in relations:
            rel_type = relation.get('type', relation.get('relation_type', 'UNKNOWN'))
            type_counts[rel_type] += 1
        
        return dict(type_counts)
    
    def hop_distance_coverage(self, 
                             query_entities: Set[str], 
                             retrieved_entities: Set[str],
                             graph_data: Optional[Dict[str, Any]] = None,
                             max_hops: int = 3) -> Dict[str, Any]:
        """
        Calculate how many hops away retrieved entities are from query entities
        
        Args:
            query_entities: Starting entities from query
            retrieved_entities: Retrieved entities
            graph_data: Graph structure
            max_hops: Maximum hops to consider
            
        Returns:
            Hop distance distribution
        """
        if graph_data is None:
            graph_data = self.graph_data
        
        if graph_data is None:
            return {'error': 'No graph data available'}
        
        # Build adjacency list
        adj = defaultdict(set)
        for edge in graph_data['edges']:
            adj[edge['source']].add(edge['target'])
            adj[edge['target']].add(edge['source'])  # Bidirectional
        
        # BFS from query entities
        hop_counts = defaultdict(int)
        
        for start_entity in query_entities:
            if start_entity not in graph_data['nodes']:
                continue
            
            visited = {start_entity}
            queue = [(start_entity, 0)]
            
            while queue:
                node, hops = queue.pop(0)
                
                if node in retrieved_entities:
                    hop_counts[hops] += 1
                
                if hops < max_hops:
                    for neighbor in adj.get(node, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, hops + 1))
        
        return {
            'hop_distribution': dict(hop_counts),
            'avg_hops': sum(h * c for h, c in hop_counts.items()) / sum(hop_counts.values()) if hop_counts else 0,
            'max_hops_found': max(hop_counts.keys()) if hop_counts else 0,
            'entities_within_1_hop': hop_counts.get(1, 0),
            'entities_within_2_hops': hop_counts.get(1, 0) + hop_counts.get(2, 0)
        }
    
    def calculate_all_graph_metrics(self, 
                                   retrieved_data: Dict[str, Any],
                                   ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive graph-based metrics
        
        Args:
            retrieved_data: Retrieved entities, relations, and subgraph
            ground_truth: Optional ground truth for comparison
            
        Returns:
            Dictionary with all graph metrics
        """
        results = {}
        
        # Basic retrieval counts
        results['num_entities_retrieved'] = len(retrieved_data.get('entities', []))
        results['num_relations_retrieved'] = len(retrieved_data.get('relations', []))
        
        # Entity and relation type distributions
        if retrieved_data.get('entities'):
            results['entity_type_distribution'] = self.entity_type_distribution(
                retrieved_data['entities']
            )
        
        if retrieved_data.get('relations'):
            results['relation_type_distribution'] = self.relation_type_distribution(
                retrieved_data['relations']
            )
        
        # Coverage metrics (if ground truth provided)
        if ground_truth:
            if 'entities' in ground_truth:
                results['entity_coverage'] = self.entity_coverage_score(
                    [e.get('id', str(e)) for e in retrieved_data.get('entities', [])],
                    set(ground_truth['entities'])
                )
            
            if 'relations' in ground_truth:
                retrieved_rels = [
                    (r.get('source'), r.get('type'), r.get('target'))
                    for r in retrieved_data.get('relations', [])
                ]
                results['relation_coverage'] = self.relation_coverage_score(
                    retrieved_rels,
                    set(ground_truth['relations'])
                )
        
        # Subgraph quality
        if 'subgraph' in retrieved_data:
            results['subgraph_quality'] = self.graph_traversal_quality(
                query=retrieved_data.get('query', ''),
                retrieved_subgraph=retrieved_data['subgraph']
            )
        
        return results


# Testing and example usage
if __name__ == "__main__":
    print("🧪 Testing Graph Metrics")
    
    metrics = GraphMetrics()
    
    # Test entity coverage
    retrieved_entities = ["Paris", "France", "Europe", "City"]
    ground_truth_entities = {"Paris", "France", "Capital"}
    
    coverage = metrics.entity_coverage_score(retrieved_entities, ground_truth_entities)
    print(f"\n📊 Entity Coverage:")
    for metric, score in coverage.items():
        print(f"  {metric}: {score:.3f}" if isinstance(score, float) else f"  {metric}: {score}")
    
    # Test relation coverage
    retrieved_rels = [
        ("Paris", "capital_of", "France"),
        ("France", "located_in", "Europe"),
        ("Paris", "is_a", "City")
    ]
    ground_truth_rels = {
        ("Paris", "capital_of", "France"),
        ("Paris", "located_in", "France")
    }
    
    rel_coverage = metrics.relation_coverage_score(retrieved_rels, ground_truth_rels)
    print(f"\n📊 Relation Coverage:")
    for metric, score in rel_coverage.items():
        print(f"  {metric}: {score:.3f}" if isinstance(score, float) else f"  {metric}: {score}")
    
    # Test entity type distribution
    entities = [
        {'id': '1', 'type': 'PERSON'},
        {'id': '2', 'type': 'LOCATION'},
        {'id': '3', 'type': 'LOCATION'},
        {'id': '4', 'type': 'ORGANIZATION'}
    ]
    
    distribution = metrics.entity_type_distribution(entities)
    print(f"\n📊 Entity Type Distribution:")
    for entity_type, count in distribution.items():
        print(f"  {entity_type}: {count}")
