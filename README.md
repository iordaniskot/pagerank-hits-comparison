# PageRank vs HITS Algorithm Comparison

This project implements and compares the PageRank and HITS (Hyperlink-Induced Topic Search) algorithms for analyzing web graphs and networks. It includes custom implementations of both algorithms and comprehensive performance analysis.

## Project Structure

```
├── graph_theory.py           # Main implementation and comparison
├── dataset_downloader.py     # Utilities for real dataset analysis  
├── activate_env.sh          # Virtual environment activation script
├── requirements.txt         # Python dependencies
├── graph_theory_env/        # Virtual environment directory
└── README.md               # This file
```

## Features

### Algorithm Implementations
- **Custom PageRank Algorithm**: Power iteration method with damping factor
- **Custom HITS Algorithm**: Mutually reinforcing authorities and hubs calculation
- **Performance Comparison**: Timing and accuracy comparison with NetworkX implementations

### Analysis Capabilities
- Correlation analysis between PageRank and HITS scores
- Visualization of score distributions and top nodes
- Performance benchmarking on different graph sizes
- Support for both synthetic and real-world datasets

### Dataset Support
- Synthetic graph generation (Barabási-Albert model)
- Real dataset downloading from networkrepository.com
- Support for various graph formats (edge lists, adjacency matrices)

## Quick Start

### 1. Environment Setup

```bash
# Activate the virtual environment
source activate_env.sh

# Or manually:
source graph_theory_env/bin/activate
```

### 2. Basic Analysis

```python
# Run the complete comparison analysis
python graph_theory.py
```

This will:
- Test algorithms on a small sample graph
- Generate a larger synthetic dataset for testing
- Create comparison visualizations
- Save detailed analysis reports

### 3. Real Dataset Analysis

```python
# Download and analyze real datasets
python dataset_downloader.py

# Or programmatically:
from dataset_downloader import analyze_real_dataset
analyze_real_dataset('ca-GrQc')  # General Relativity collaboration network
```

## Algorithm Details

### PageRank
PageRank measures the importance of nodes based on the link structure of the graph. It uses the formula:

```
PR(i) = (1-α)/N + α * Σ(PR(j)/L(j))
```

Where:
- `α` is the damping factor (default: 0.85)
- `N` is the total number of nodes
- `L(j)` is the out-degree of node j

**Key Features:**
- Single importance score per node
- Global authority measurement
- Handles dangling nodes appropriately
- Used by Google's original search algorithm

### HITS (Authorities and Hubs)
HITS distinguishes between two types of nodes:
- **Authorities**: Nodes with good content (high in-degree from good hubs)
- **Hubs**: Nodes that link to good authorities (high out-degree to good authorities)

**Update Rules:**
```
Authority(i) = Σ Hub(j) for all j linking to i
Hub(i) = Σ Authority(j) for all j that i links to
```

**Key Features:**
- Two scores per node (authority and hub)
- Mutually reinforcing relationship
- Good for topic-specific search
- Captures different aspects of node importance

## Performance Analysis

The implementation provides comprehensive performance analysis including:

### Timing Comparison
- Custom implementation vs NetworkX
- Scalability across different graph sizes
- Convergence analysis

### Correlation Analysis
- PageRank vs Authority scores
- PageRank vs Hub scores  
- Authority vs Hub scores

### Top Nodes Analysis
- Identification of most important nodes by each metric
- Comparison of ranking differences
- Visualization of score distributions

## Available Datasets

The project includes access to several real-world datasets from networkrepository.com:

1. **ca-GrQc**: General Relativity collaboration network (5,242 nodes, 14,496 edges)
2. **wiki-Vote**: Wikipedia voting network (7,115 nodes, 103,689 edges)
3. **p2p-Gnutella04**: Gnutella peer-to-peer network (10,876 nodes, 39,994 edges)
4. **web-NotreDame**: Web graph of Notre Dame university (325,729 nodes, 1,497,134 edges)

## Output Files

The analysis generates several output files:

### Reports
- `pagerank_hits_comparison_report.txt`: Detailed analysis report
- `{dataset_name}_analysis_report.txt`: Dataset-specific analysis

### Visualizations
- `sample_graph_comparison.png`: Small graph analysis plots
- `large_graph_comparison.png`: Large graph analysis plots
- `{dataset_name}_comparison.png`: Dataset-specific visualizations

### Data Files
- `sample_network.txt`: Generated synthetic network
- `requirements.txt`: Python dependencies

## Usage Examples

### Custom Graph Analysis

```python
from graph_theory import GraphAnalyzer
import networkx as nx

# Create analyzer
analyzer = GraphAnalyzer()

# Load your own graph
analyzer.load_graph_from_edgelist("your_graph.txt", directed=True)

# Or create from NetworkX graph
G = nx.erdos_renyi_graph(100, 0.1, directed=True)
analyzer.graph = G

# Run analysis
results = analyzer.compare_with_networkx()
correlations = analyzer.analyze_correlations(results)
report = analyzer.generate_performance_report(results, correlations)
print(report)

# Create visualizations
analyzer.plot_comparison(results, "my_analysis.png")
```

### Parameter Tuning

```python
# Test different damping factors for PageRank
damping_factors = [0.1, 0.5, 0.85, 0.95]

for alpha in damping_factors:
    pr_scores = analyzer.pagerank_custom(alpha=alpha)
    print(f"Alpha {alpha}: Top node score = {max(pr_scores.values()):.4f}")
```

### Batch Analysis

```python
from dataset_downloader import get_sample_datasets, analyze_real_dataset

# Analyze all available datasets
datasets = get_sample_datasets()
for dataset_name in datasets.keys():
    print(f"Analyzing {dataset_name}...")
    analyze_real_dataset(dataset_name)
```

## Requirements

- Python 3.7+
- NetworkX 3.4+
- NumPy 1.21+
- Matplotlib 3.5+
- Pandas 1.3+
- SciPy 1.7+
- Seaborn 0.11+
- Requests 2.25+

## Installation

All dependencies are included in the virtual environment. To set up from scratch:

```bash
python3 -m venv graph_theory_env
source graph_theory_env/bin/activate
pip install -r requirements.txt
```

## Algorithm Comparison Summary

| Aspect | PageRank | HITS |
|--------|----------|------|
| **Scores per node** | 1 (importance) | 2 (authority + hub) |
| **Scope** | Global importance | Local topic relevance |
| **Best for** | General web search | Topic-specific search |
| **Computation** | Single eigenvector | Two mutually reinforcing vectors |
| **Damping** | Uses damping factor | No damping |
| **Convergence** | Usually slower | Often faster |

## Research Applications

This implementation is suitable for:

- **Web Graph Analysis**: Understanding link structures and authority
- **Social Network Analysis**: Identifying influential users and information spreaders
- **Citation Networks**: Finding important papers and prolific authors
- **Transportation Networks**: Identifying critical hubs and routes
- **Biological Networks**: Understanding protein interactions and regulatory networks

## Future Extensions

Potential improvements and extensions:

1. **Personalized PageRank**: Topic-sensitive importance calculation
2. **Weighted HITS**: Incorporating edge weights in calculations  
3. **Temporal Analysis**: Studying algorithm behavior over time
4. **Memory Optimization**: Sparse matrix implementations for large graphs
5. **Parallel Computing**: Multi-threaded implementations for scalability

## References

1. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web.
2. Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked environment. Journal of the ACM, 46(5), 604-632.
3. NetworkX Documentation: https://networkx.org/
4. Network Repository: http://networkrepository.com/


