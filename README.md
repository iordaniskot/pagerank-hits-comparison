# PageRank vs HITS Algorithm Comparison

This project implements and compares the PageRank and HITS (Hyperlink-Induced Topic Search) algorithms for analyzing web graphs and networks. It includes custom implementations of both algorithms and comprehensive performance analysis.

> **🎯 Quick Start Recommendation:** For the best experience and comprehensive results, run `python run_complete_analysis.py` after setup. This single command provides a complete demonstration of all features in 1-2 minutes.

## Project Structure

```
├── graph_theory.py           # Main implementation and comparison
├── dataset_downloader.py     # Utilities for real dataset analysis  
├── run_complete_analysis.py  # Complete analysis suite runner
├── cleanup_utility.py        # Standalone cleanup utility
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
- **Automatic cleanup of temporary files** after analysis completion

### Dataset Support
- Synthetic graph generation (Barabási-Albert model)
- Real dataset downloading from networkrepository.com
- Support for various graph formats (edge lists, Matrix Market, adjacency matrices)
- **Automatic cleanup of downloaded and extracted files**

## Quick Start

### 1. Environment Setup

```bash
# Activate the virtual environment
source activate_env.sh

# Or manually:
source graph_theory_env/bin/activate
```

### 2. **Recommended: Complete Analysis Suite** ⭐

**Run the complete analysis with a single command:**

```bash
python run_complete_analysis.py
```

This is the **recommended way** to experience the full capabilities of this project. It will:
- Run analysis on synthetic graphs (small and large)
- Download and analyze multiple real-world datasets
- Generate comprehensive comparison reports and visualizations
- Automatically clean up temporary files
- Create a complete suite of results in 2-5 minutes

**Perfect for:**
- Academic submissions and demonstrations
- Getting comprehensive results quickly
- Comparing multiple datasets at once
- Automatic file management

### 3. Individual Component Analysis

If you prefer to run components separately:

**Basic Algorithm Testing:**
```bash
python graph_theory.py
```

**Real Dataset Analysis:**
```bash
python dataset_downloader.py

# Or programmatically:
from dataset_downloader import analyze_real_dataset
analyze_real_dataset('ca-GrQc')  # General Relativity collaboration network
```

## Why Use run_complete_analysis.py? ⭐

The `run_complete_analysis.py` script is the **ideal choice** for academic presentations, coursework submissions, and comprehensive demonstrations because it:

### 🚀 **Comprehensive Coverage**
- Tests both small and large-scale scenarios
- Analyzes 4-5 different real-world network types
- Demonstrates scalability across different graph sizes
- Shows algorithm behavior on various network topologies

### 📊 **Professional Results**
- Generates publication-ready visualizations
- Creates detailed comparison reports
- Provides statistical correlation analysis
- Includes performance benchmarks

### 🧹 **Zero Maintenance**
- Automatically downloads required datasets
- Cleans up temporary files after analysis
- Manages memory efficiently
- Leaves only the important results

### ⏱️ **Time Efficient**
- Complete analysis in 2-5 minutes
- Single command execution
- Parallel processing where possible
- No manual intervention required

**Perfect for academic environments** where you need to demonstrate algorithm understanding, compare performance metrics, and present professional results quickly.

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

### File Cleanup

The project includes automatic cleanup of temporary files created during analysis:

```python
# Manual cleanup of specific dataset files
from dataset_downloader import cleanup_temporary_files
cleanup_temporary_files('soc-karate')  # Clean files for specific dataset

# Clean all temporary files
cleanup_temporary_files()  # Clean all temporary files
```

**Standalone Cleanup Utility:**

```bash
# List temporary files without removing them
python cleanup_utility.py --list

# Clean all temporary files
python cleanup_utility.py --all

# Clean files for specific dataset
python cleanup_utility.py --dataset soc-karate

# Interactive mode
python cleanup_utility.py
```

**Automatically cleaned file types:**
- `.mtx` - Matrix Market format files
- `.edges` - Edge list files  
- `.zip` - Downloaded archive files
- `*_processed.txt` - Processed SNAP dataset files

**Note:** Analysis reports (`*_analysis_report.txt`) and visualizations (`*_comparison.png`) are preserved.
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


## References

1. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web.
2. Kleinberg, J. M. (1999). Authoritative sources in a hyperlinked environment. Journal of the ACM, 46(5), 604-632.
3. NetworkX Documentation: https://networkx.org/
4. Network Repository: http://networkrepository.com/


