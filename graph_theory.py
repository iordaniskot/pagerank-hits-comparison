import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import requests
import zipfile
import os
from typing import Dict, Tuple, List
import seaborn as sns

class GraphAnalyzer:
    """
    A class to implement and compare PageRank and HITS algorithms
    """
    
    def __init__(self, graph: nx.Graph = None):
        """
        Initialize the analyzer with a graph
        """
        self.graph = graph
        self.pagerank_scores = None
        self.hits_authorities = None
        self.hits_hubs = None
        
    def load_graph_from_edgelist(self, filepath: str, directed: bool = True) -> nx.Graph:
        """
        Load graph from edge list file
        """
        try:
            if directed:
                self.graph = nx.read_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
            else:
                self.graph = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
            print(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return self.graph
        except Exception as e:
            print(f"Error loading graph: {e}")
            return None
    
    def create_sample_graph(self) -> nx.DiGraph:
        """
        Create a sample directed graph for testing
        """
        G = nx.DiGraph()
        # Create a small web-like graph
        edges = [
            (1, 2), (1, 3), (2, 3), (2, 4), 
            (3, 1), (3, 4), (4, 2), (4, 5),
            (5, 1), (5, 3), (6, 1), (6, 5)
        ]
        G.add_edges_from(edges)
        self.graph = G
        return G
    
    def pagerank_custom(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict:
        """
        Custom implementation of PageRank algorithm
        
        Args:
            alpha: Damping factor (probability of following links)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary of node: pagerank_score
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        # Convert to directed graph if undirected
        if not self.graph.is_directed():
            G = self.graph.to_directed()
        else:
            G = self.graph
        
        nodes = list(G.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(G, nodelist=nodes, dtype=float)
        
        # Get out-degrees
        out_degrees = np.array([G.out_degree(node) for node in nodes], dtype=float)
        
        # Handle dangling nodes (nodes with no outgoing edges)
        dangling_nodes = out_degrees == 0
        out_degrees[dangling_nodes] = 1
        
        # Create transition matrix
        transition_matrix = adj_matrix.T / out_degrees
        
        # Initialize PageRank values
        pagerank = np.ones(n) / n
        
        # Power iteration
        for iteration in range(max_iter):
            prev_pagerank = pagerank.copy()
            
            # PageRank formula: PR(i) = (1-α)/N + α * Σ(PR(j)/L(j)) for all j linking to i
            pagerank = (1 - alpha) / n + alpha * transition_matrix.dot(pagerank)
            
            # Handle dangling nodes - distribute their PageRank equally
            dangling_sum = alpha * np.sum(prev_pagerank[dangling_nodes]) / n
            pagerank += dangling_sum
            
            # Check convergence
            if np.sum(np.abs(pagerank - prev_pagerank)) < tol:
                print(f"PageRank converged after {iteration + 1} iterations")
                break
        
        # Create result dictionary
        self.pagerank_scores = dict(zip(nodes, pagerank))
        return self.pagerank_scores
    
    def hits_custom(self, max_iter: int = 100, tol: float = 1e-6) -> Tuple[Dict, Dict]:
        """
        Custom implementation of HITS (Hyperlink-Induced Topic Search) algorithm
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (authorities_dict, hubs_dict)
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        # Convert to directed graph if undirected
        if not self.graph.is_directed():
            G = self.graph.to_directed()
        else:
            G = self.graph
        
        nodes = list(G.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}, {}
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(G, nodelist=nodes, dtype=float).toarray()
        
        # Initialize authority and hub scores
        authorities = np.ones(n)
        hubs = np.ones(n)
        
        # Power iteration
        for iteration in range(max_iter):
            prev_authorities = authorities.copy()
            prev_hubs = hubs.copy()
            
            # Update authority scores: auth(i) = Σ hub(j) for all j linking to i
            authorities = adj_matrix.T.dot(hubs)
            
            # Update hub scores: hub(i) = Σ auth(j) for all j that i links to
            hubs = adj_matrix.dot(authorities)
            
            # Normalize scores
            auth_norm = np.linalg.norm(authorities)
            hub_norm = np.linalg.norm(hubs)
            
            if auth_norm > 0:
                authorities = authorities / auth_norm
            if hub_norm > 0:
                hubs = hubs / hub_norm
            
            # Check convergence
            auth_diff = np.sum(np.abs(authorities - prev_authorities))
            hub_diff = np.sum(np.abs(hubs - prev_hubs))
            
            if auth_diff < tol and hub_diff < tol:
                print(f"HITS converged after {iteration + 1} iterations")
                break
        
        # Create result dictionaries
        self.hits_authorities = dict(zip(nodes, authorities))
        self.hits_hubs = dict(zip(nodes, hubs))
        
        return self.hits_authorities, self.hits_hubs
    
    def compare_with_networkx(self) -> Dict:
        """
        Compare custom implementations with NetworkX implementations
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        results = {}
        
        # Compare PageRank
        start_time = time.time()
        custom_pr = self.pagerank_custom()
        custom_pr_time = time.time() - start_time
        
        start_time = time.time()
        nx_pr = nx.pagerank(self.graph, alpha=0.85, max_iter=100, tol=1e-6)
        nx_pr_time = time.time() - start_time
        
        # Compare HITS
        start_time = time.time()
        custom_auth, custom_hub = self.hits_custom()
        custom_hits_time = time.time() - start_time
        
        start_time = time.time()
        nx_hits = nx.hits(self.graph, max_iter=100, tol=1e-6)
        nx_hits_time = time.time() - start_time
        
        results = {
            'custom_pagerank': custom_pr,
            'networkx_pagerank': nx_pr,
            'custom_authorities': custom_auth,
            'custom_hubs': custom_hub,
            'networkx_authorities': nx_hits[1],
            'networkx_hubs': nx_hits[0],
            'custom_pr_time': custom_pr_time,
            'nx_pr_time': nx_pr_time,
            'custom_hits_time': custom_hits_time,
            'nx_hits_time': nx_hits_time
        }
        
        return results
    
    def analyze_correlations(self, results: Dict) -> Dict:
        """
        Analyze correlations between PageRank and HITS scores
        """
        nodes = list(self.graph.nodes())
        
        # Extract scores for correlation analysis
        pagerank_scores = [results['custom_pagerank'][node] for node in nodes]
        authority_scores = [results['custom_authorities'][node] for node in nodes]
        hub_scores = [results['custom_hubs'][node] for node in nodes]
        
        # Calculate correlations
        pr_auth_corr = np.corrcoef(pagerank_scores, authority_scores)[0, 1]
        pr_hub_corr = np.corrcoef(pagerank_scores, hub_scores)[0, 1]
        auth_hub_corr = np.corrcoef(authority_scores, hub_scores)[0, 1]
        
        return {
            'pagerank_authority_correlation': pr_auth_corr,
            'pagerank_hub_correlation': pr_hub_corr,
            'authority_hub_correlation': auth_hub_corr
        }
    
    def plot_comparison(self, results: Dict, save_path: str = None):
        """
        Create visualizations comparing PageRank and HITS
        """
        nodes = list(self.graph.nodes())
        
        # Extract scores
        pagerank_scores = [results['custom_pagerank'][node] for node in nodes]
        authority_scores = [results['custom_authorities'][node] for node in nodes]
        hub_scores = [results['custom_hubs'][node] for node in nodes]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: PageRank vs Authority scores
        axes[0, 0].scatter(pagerank_scores, authority_scores, alpha=0.7)
        axes[0, 0].set_xlabel('PageRank Score')
        axes[0, 0].set_ylabel('Authority Score')
        axes[0, 0].set_title('PageRank vs Authority Scores')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: PageRank vs Hub scores
        axes[0, 1].scatter(pagerank_scores, hub_scores, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('PageRank Score')
        axes[0, 1].set_ylabel('Hub Score')
        axes[0, 1].set_title('PageRank vs Hub Scores')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Authority vs Hub scores
        axes[1, 0].scatter(authority_scores, hub_scores, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Authority Score')
        axes[1, 0].set_ylabel('Hub Score')
        axes[1, 0].set_title('Authority vs Hub Scores')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Top nodes comparison
        top_n = min(10, len(nodes))
        
        # Get top nodes for each metric
        top_pr = sorted(results['custom_pagerank'].items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_auth = sorted(results['custom_authorities'].items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_hub = sorted(results['custom_hubs'].items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        x_pos = np.arange(top_n)
        width = 0.25
        
        axes[1, 1].bar(x_pos - width, [score for _, score in top_pr], width, label='PageRank', alpha=0.8)
        axes[1, 1].bar(x_pos, [score for _, score in top_auth], width, label='Authority', alpha=0.8)
        axes[1, 1].bar(x_pos + width, [score for _, score in top_hub], width, label='Hub', alpha=0.8)
        
        axes[1, 1].set_xlabel('Top Nodes (by PageRank)')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title(f'Top {top_n} Nodes Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([str(node) for node, _ in top_pr], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(self, results: Dict, correlations: Dict) -> str:
        """
        Generate a comprehensive performance report
        """
        report = []
        report.append("=" * 80)
        report.append("PAGERANK vs HITS ALGORITHM COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Graph statistics
        report.append("GRAPH STATISTICS:")
        report.append(f"  Nodes: {self.graph.number_of_nodes()}")
        report.append(f"  Edges: {self.graph.number_of_edges()}")
        report.append(f"  Directed: {self.graph.is_directed()}")
        report.append(f"  Average degree: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.2f}")
        report.append("")
        
        # Performance comparison
        report.append("PERFORMANCE COMPARISON:")
        report.append(f"  Custom PageRank time: {results['custom_pr_time']:.4f} seconds")
        report.append(f"  NetworkX PageRank time: {results['nx_pr_time']:.4f} seconds")
        report.append(f"  Custom HITS time: {results['custom_hits_time']:.4f} seconds")
        report.append(f"  NetworkX HITS time: {results['nx_hits_time']:.4f} seconds")
        report.append("")
        
        # Correlation analysis
        report.append("CORRELATION ANALYSIS:")
        report.append(f"  PageRank vs Authority: {correlations['pagerank_authority_correlation']:.4f}")
        report.append(f"  PageRank vs Hub: {correlations['pagerank_hub_correlation']:.4f}")
        report.append(f"  Authority vs Hub: {correlations['authority_hub_correlation']:.4f}")
        report.append("")
        
        # Top nodes analysis
        top_n = min(5, len(self.graph.nodes()))
        report.append(f"TOP {top_n} NODES ANALYSIS:")
        
        top_pr = sorted(results['custom_pagerank'].items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_auth = sorted(results['custom_authorities'].items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_hub = sorted(results['custom_hubs'].items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        report.append("  PageRank Top Nodes:")
        for i, (node, score) in enumerate(top_pr, 1):
            report.append(f"    {i}. Node {node}: {score:.6f}")
        
        report.append("  Authority Top Nodes:")
        for i, (node, score) in enumerate(top_auth, 1):
            report.append(f"    {i}. Node {node}: {score:.6f}")
        
        report.append("  Hub Top Nodes:")
        for i, (node, score) in enumerate(top_hub, 1):
            report.append(f"    {i}. Node {node}: {score:.6f}")
        
        report.append("")
        report.append("ALGORITHM COMPARISON SUMMARY:")
        report.append("  PageRank:")
        report.append("    - Measures global importance based on link structure")
        report.append("    - Single score per node")
        report.append("    - Good for finding authoritative pages")
        report.append("    - Used by Google's original search algorithm")
        report.append("")
        report.append("  HITS (Authorities and Hubs):")
        report.append("    - Distinguishes between authorities (good content) and hubs (good links)")
        report.append("    - Two scores per node")
        report.append("    - Good for topic-specific search")
        report.append("    - Mutually reinforcing relationship between authorities and hubs")
        report.append("")
        
        return "\n".join(report)

def download_sample_dataset():
    """
    Download a sample dataset for testing
    """
    # For demonstration, we'll create a synthetic dataset
    # In practice, you would download from networkrepository.com
    print("Creating sample dataset...")
    
    # Create a larger sample graph
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    G = G.to_directed()
    
    # Add some additional structure
    for i in range(10):
        for j in range(i+10, min(i+20, 100)):
            if np.random.random() < 0.1:
                G.add_edge(i, j)
    
    # Save as edge list
    nx.write_edgelist(G, "sample_network.txt", data=False)
    print("Sample dataset created: sample_network.txt")
    return "sample_network.txt"

def main():
    """
    Main function to run the comparison
    """
    print("PageRank vs HITS Algorithm Comparison")
    print("====================================")
    
    # Initialize analyzer
    analyzer = GraphAnalyzer()
    
    # Option 1: Use sample graph
    print("\n1. Testing with sample graph...")
    analyzer.create_sample_graph()
    
    # Run comparison
    results = analyzer.compare_with_networkx()
    correlations = analyzer.analyze_correlations(results)
    
    # Generate report
    report = analyzer.generate_performance_report(results, correlations)
    print(report)
    
    # Plot comparison
    analyzer.plot_comparison(results, "sample_graph_comparison.png")
    
    # Option 2: Test with larger synthetic dataset
    print("\n2. Testing with larger synthetic dataset...")
    dataset_file = download_sample_dataset()
    analyzer.load_graph_from_edgelist(dataset_file, directed=True)
    
    # Run comparison on larger dataset
    results_large = analyzer.compare_with_networkx()
    correlations_large = analyzer.analyze_correlations(results_large)
    
    # Generate report for larger dataset
    report_large = analyzer.generate_performance_report(results_large, correlations_large)
    print("\n" + "="*50)
    print("LARGE DATASET RESULTS:")
    print("="*50)
    print(report_large)
    
    # Plot comparison for larger dataset
    analyzer.plot_comparison(results_large, "large_graph_comparison.png")
    
    # Save detailed results
    print("\nSaving detailed results...")
    with open("pagerank_hits_comparison_report.txt", "w") as f:
        f.write("SMALL SAMPLE GRAPH RESULTS:\n")
        f.write("="*50 + "\n")
        f.write(report)
        f.write("\n\n")
        f.write("LARGE SYNTHETIC DATASET RESULTS:\n")
        f.write("="*50 + "\n")
        f.write(report_large)
    
    print("Analysis complete! Check the generated files:")
    print("- pagerank_hits_comparison_report.txt")
    print("- sample_graph_comparison.png")
    print("- large_graph_comparison.png")

if __name__ == "__main__":
    main()